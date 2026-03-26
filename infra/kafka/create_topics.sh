#!/bin/bash
# Wait for Kafka to be fully ready
sleep 10

KAFKA_BROKER="kafka:29092"

echo "Creating Kafka topics..."

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.patient.events \
  --partitions 6 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.context.events \
  --partitions 2 \
  --replication-factor 1 \
  --config retention.ms=259200000

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.feature.windows \
  --partitions 6 \
  --replication-factor 1 \
  --config retention.ms=172800000

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.predictions \
  --partitions 6 \
  --replication-factor 1 \
  --config retention.ms=2592000000

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.alerts \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=7776000000

kafka-topics --create --if-not-exists \
  --bootstrap-server $KAFKA_BROKER \
  --topic ed.dead.letter \
  --partitions 2 \
  --replication-factor 1 \
  --config retention.ms=2592000000

echo "All Kafka topics created successfully."
kafka-topics --list --bootstrap-server $KAFKA_BROKER
