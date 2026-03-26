"""
Reads ed.context.events from Kafka, normalizes context signals,
and writes enriched context features to Cassandra for model use.
"""
import os
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, BooleanType, IntegerType
)

logger = logging.getLogger(__name__)

KAFKA_SERVERS      = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CASSANDRA_KEYSPACE = "ed_forecasting"
WATERMARK_DURATION = "30 minutes"
WINDOW_DURATION    = "30 minutes"

# ─── Schema for context events ────────────────────────────────────────────────
CONTEXT_SCHEMA = StructType([
    StructField("event_id",           StringType(),    True),
    StructField("event_type",         StringType(),    True),
    StructField("hospital_id",        StringType(),    True),
    StructField("timestamp",          TimestampType(), True),
    # Weather
    StructField("temperature_c",      DoubleType(),    True),
    StructField("precipitation",      DoubleType(),    True),
    StructField("is_severe",          BooleanType(),   True),
    StructField("weather_score",      DoubleType(),    True),
    # Flu
    StructField("flu_index",          DoubleType(),    True),
    StructField("trend",              StringType(),    True),
    # Traffic
    StructField("congestion_score",   DoubleType(),    True),
    StructField("avg_speed_kmh",      DoubleType(),    True),
])


def run_context_stream(spark: SparkSession):
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", "ed.context.events")
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw
        .select(F.from_json(F.col("value").cast("string"), CONTEXT_SCHEMA).alias("d"))
        .select("d.*")
        .withWatermark("timestamp", WATERMARK_DURATION)
    )

    # ── Aggregate context per hospital per 30-min window ─────────────────────
    context_agg = (
        parsed
        .groupBy(
            F.window("timestamp", WINDOW_DURATION),
            "hospital_id",
        )
        .agg(
            F.avg("weather_score").alias("weather_score"),
            F.avg("flu_index").alias("flu_index"),
            F.avg("congestion_score").alias("traffic_score"),
            F.max(F.col("is_severe").cast("integer")).alias("is_severe_weather"),
        )
        .withColumn("window_end", F.col("window.end"))
        .drop("window")
    )

    def write_context(batch_df: DataFrame, batch_id: int):
        if batch_df.isEmpty():
            return
        logger.info(f"Context batch {batch_id}: {batch_df.count()} rows")
        # Write to Kafka for joining with patient features
        (
            batch_df
            .select(
                F.col("hospital_id").alias("key"),
                F.to_json(F.struct("*")).alias("value"),
            )
            .write
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_SERVERS)
            .option("topic", "ed.context.events.enriched")
            .save()
        )

    query = (
        context_agg
        .writeStream
        .foreachBatch(write_context)
        .outputMode("update")
        .option("checkpointLocation", "/tmp/ed_checkpoint/context_stream")
        .trigger(processingTime="60 seconds")
        .start()
    )
    return query
