"""
Base Kafka producer with delivery report callbacks, dead-letter routing,
idempotent delivery, and structured error logging.
"""
import logging
import os
from confluent_kafka import Producer, KafkaError

logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEAD_LETTER_TOPIC = "ed.dead.letter"


class BaseProducer:
    def __init__(self, client_id: str):
        self._producer = Producer({
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "client.id": client_id,
            "acks": "all",
            "enable.idempotence": True,
            "max.in.flight.requests.per.connection": 5,
            "retries": 5,
            "retry.backoff.ms": 300,
            "compression.type": "lz4",
        })
        self._failed_count = 0

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Delivery failed for topic={msg.topic()} key={msg.key()}: {err}")
            self._failed_count += 1
            if self._failed_count % 100 == 0:
                logger.warning(f"Total delivery failures so far: {self._failed_count}")
            # Route to dead-letter topic (best-effort)
            try:
                self._producer.produce(
                    topic=DEAD_LETTER_TOPIC,
                    key=msg.key(),
                    value=msg.value(),
                    headers={"original_topic": msg.topic(), "error": str(err)},
                )
            except Exception as dlq_err:
                logger.error(f"Failed to publish to DLQ: {dlq_err}")
        else:
            logger.debug(f"Delivered to {msg.topic()}[{msg.partition()}] offset={msg.offset()}")

    def _produce(self, topic: str, key: str, value: bytes) -> None:
        try:
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=value,
                callback=self._delivery_report,
            )
            self._producer.poll(0)  # trigger callbacks without blocking
        except BufferError:
            logger.warning("Producer queue full, flushing...")
            self._producer.flush(timeout=5)
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=value,
                callback=self._delivery_report,
            )

    def flush(self, timeout: float = 10.0):
        self._producer.flush(timeout=timeout)

    def close(self):
        self.flush()
        logger.info("Producer closed.")
