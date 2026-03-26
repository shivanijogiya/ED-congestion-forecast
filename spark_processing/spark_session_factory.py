"""
SparkSession factory with Kafka and Cassandra connectors configured.
"""
import os
from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "EDForecast") -> SparkSession:
    cassandra_host = os.getenv("CASSANDRA_HOST", "localhost")
    kafka_servers  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        # Kafka connector
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "com.datastax.spark:spark-cassandra-connector_2.12:3.4.1,"
            "org.apache.spark:spark-avro_2.12:3.5.0",
        )
        # Cassandra connector
        .config("spark.cassandra.connection.host", cassandra_host)
        .config("spark.cassandra.connection.port", "9042")
        # Streaming tuning
        .config("spark.streaming.kafka.maxRatePerPartition", "1000")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/ed_checkpoint")
        # Memory
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
