"""
Reads ed.patient.events from Kafka and computes stateful per-department
occupancy features using Spark Structured Streaming.

Outputs per-department feature rows every 10-minute micro-batch to:
  - Cassandra: feature_windows table
  - Kafka:     ed.feature.windows topic (for downstream joining)
"""
import os
import json
import math
import logging
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, IntegerType, BooleanType
)

logger = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CASSANDRA_KEYSPACE = "ed_forecasting"
WATERMARK_DURATION = "10 minutes"
WINDOW_DURATION = "10 minutes"
SLIDE_DURATION  = "10 minutes"


# ─── Input schema for patient events ─────────────────────────────────────────
PATIENT_EVENT_SCHEMA = StructType([
    StructField("event_id",    StringType(),    True),
    StructField("event_type",  StringType(),    True),
    StructField("hospital_id", StringType(),    True),
    StructField("dept_id",     StringType(),    True),
    StructField("patient_id",  StringType(),    True),
    StructField("acuity",      IntegerType(),   True),
    StructField("arrival_ts",  TimestampType(), True),
    StructField("discharge_ts",TimestampType(), True),
    StructField("transfer_ts", TimestampType(), True),
    StructField("source_dept_id",    StringType(), True),
    StructField("target_dept_id",    StringType(), True),
    StructField("expected_los_minutes",  DoubleType(), True),
    StructField("actual_los_minutes",    DoubleType(), True),
])


def run_patient_stream(spark: SparkSession):
    # ── Read from Kafka ───────────────────────────────────────────────────────
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", "ed.patient.events")
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ── Parse JSON payload ────────────────────────────────────────────────────
    parsed = (
        raw_stream
        .select(F.from_json(F.col("value").cast("string"), PATIENT_EVENT_SCHEMA).alias("data"))
        .select("data.*")
    )

    # Unify timestamp: use event-specific timestamp column
    with_ts = parsed.withColumn(
        "event_ts",
        F.coalesce(
            F.col("arrival_ts"),
            F.col("discharge_ts"),
            F.col("transfer_ts"),
        )
    ).withWatermark("event_ts", WATERMARK_DURATION)

    # ── Arrival aggregation ───────────────────────────────────────────────────
    arrivals = (
        with_ts
        .filter(F.col("event_type") == "arrival")
        .groupBy(
            F.window("event_ts", WINDOW_DURATION, SLIDE_DURATION),
            "hospital_id", "dept_id"
        )
        .agg(
            F.count("*").alias("arrival_count"),
            F.avg("expected_los_minutes").alias("avg_expected_los"),
            (F.sum(F.when(F.col("acuity") <= 2, 1).otherwise(0)) / F.count("*")).alias("severity_index"),
        )
    )

    # ── Discharge aggregation ─────────────────────────────────────────────────
    discharges = (
        with_ts
        .filter(F.col("event_type") == "discharge")
        .groupBy(
            F.window("event_ts", WINDOW_DURATION, SLIDE_DURATION),
            "hospital_id", "dept_id"
        )
        .agg(
            F.count("*").alias("discharge_count"),
            F.avg("actual_los_minutes").alias("avg_actual_los"),
        )
    )

    # ── Join arrivals + discharges on window + location ───────────────────────
    occupancy = (
        arrivals
        .join(
            discharges,
            on=["window", "hospital_id", "dept_id"],
            how="left"
        )
        .withColumn("discharge_count", F.coalesce(F.col("discharge_count"), F.lit(0)))
        .withColumn("window_end", F.col("window.end"))
        .withColumn("arrival_rate", F.col("arrival_count") / F.lit(1.0))  # per window
        .withColumn("hour_sin", F.sin(2 * math.pi * F.hour("window_end") / 24))
        .withColumn("hour_cos", F.cos(2 * math.pi * F.hour("window_end") / 24))
        .select(
            "hospital_id", "dept_id", "window_end",
            "arrival_count", "discharge_count", "arrival_rate",
            "avg_expected_los", "avg_actual_los",
            F.coalesce("severity_index", F.lit(0.0)).alias("severity_index"),
            "hour_sin", "hour_cos",
        )
    )

    # ── Write to Cassandra ────────────────────────────────────────────────────
    def write_to_cassandra(batch_df: DataFrame, batch_id: int):
        if batch_df.isEmpty():
            return
        (
            batch_df
            .withColumnRenamed("window_end", "window_ts")
            .write
            .format("org.apache.spark.sql.cassandra")
            .options(table="feature_windows", keyspace=CASSANDRA_KEYSPACE)
            .mode("append")
            .save()
        )
        logger.info(f"Batch {batch_id}: wrote {batch_df.count()} feature rows to Cassandra")

    # ── Write to Kafka (for downstream joining with context) ──────────────────
    def write_to_kafka(batch_df: DataFrame, batch_id: int):
        if batch_df.isEmpty():
            return
        (
            batch_df
            .select(
                F.col("hospital_id").alias("key"),
                F.to_json(F.struct("*")).alias("value"),
            )
            .write
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_SERVERS)
            .option("topic", "ed.feature.windows")
            .save()
        )

    # ── Start streaming query ─────────────────────────────────────────────────
    query = (
        occupancy
        .writeStream
        .foreachBatch(lambda df, bid: (write_to_cassandra(df, bid), write_to_kafka(df, bid)))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/ed_checkpoint/patient_stream")
        .trigger(processingTime="60 seconds")
        .start()
    )
    return query
