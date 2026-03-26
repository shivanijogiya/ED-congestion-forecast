"""
Publishes patient events (arrival, discharge, transfer) to ed.patient.events.
Partitioned by hospital_id for parallel per-hospital processing.
"""
from kafka_layer.producers.base_producer import BaseProducer
from simulation.schemas import PatientArrivalEvent, PatientDischargeEvent, PatientTransferEvent

TOPIC = "ed.patient.events"


class PatientEventProducer(BaseProducer):
    def __init__(self):
        super().__init__(client_id="patient-event-producer")

    def publish_arrival(self, event: PatientArrivalEvent) -> None:
        self._produce(
            topic=TOPIC,
            key=event.hospital_id,
            value=event.to_kafka_payload(),
        )

    def publish_discharge(self, event: PatientDischargeEvent) -> None:
        self._produce(
            topic=TOPIC,
            key=event.hospital_id,
            value=event.to_kafka_payload(),
        )

    def publish_transfer(self, event: PatientTransferEvent) -> None:
        self._produce(
            topic=TOPIC,
            key=event.hospital_id,
            value=event.to_kafka_payload(),
        )
