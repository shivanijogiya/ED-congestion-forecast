"""
Publishes external context events (weather, flu, traffic) to ed.context.events.
"""
from kafka_layer.producers.base_producer import BaseProducer
from simulation.schemas import WeatherEvent, FluIndexEvent, TrafficEvent

TOPIC = "ed.context.events"


class ContextEventProducer(BaseProducer):
    def __init__(self):
        super().__init__(client_id="context-event-producer")

    def publish_weather(self, event: WeatherEvent) -> None:
        self._produce(TOPIC, event.hospital_id, event.to_kafka_payload())

    def publish_flu_index(self, event: FluIndexEvent) -> None:
        self._produce(TOPIC, event.hospital_id, event.to_kafka_payload())

    def publish_traffic(self, event: TrafficEvent) -> None:
        self._produce(TOPIC, event.hospital_id, event.to_kafka_payload())
