"""
Pydantic schemas for all simulation event types.
Every event is validated before publishing to Kafka.
"""
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class AcuityLevel(int, Enum):
    CRITICAL       = 1   # Immediate
    EMERGENT       = 2   # < 15 min
    URGENT         = 3   # < 30 min
    LESS_URGENT    = 4   # < 60 min
    NON_URGENT     = 5   # < 120 min


class EventType(str, Enum):
    ARRIVAL    = "arrival"
    DISCHARGE  = "discharge"
    TRANSFER   = "transfer"
    WEATHER    = "weather"
    FLU_INDEX  = "flu_index"
    TRAFFIC    = "traffic"


# ─── Patient Events ───────────────────────────────────────────────────────────

class PatientArrivalEvent(BaseModel):
    event_id:    str       = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:  str       = EventType.ARRIVAL
    hospital_id: str
    dept_id:     str
    patient_id:  str       = Field(default_factory=lambda: str(uuid.uuid4()))
    acuity:      AcuityLevel
    arrival_ts:  datetime
    expected_los_minutes: float = Field(gt=0)

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class PatientDischargeEvent(BaseModel):
    event_id:     str      = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:   str      = EventType.DISCHARGE
    hospital_id:  str
    dept_id:      str
    patient_id:   str
    discharge_ts: datetime
    actual_los_minutes: float = Field(gt=0)

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class PatientTransferEvent(BaseModel):
    event_id:       str    = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:     str    = EventType.TRANSFER
    hospital_id:    str
    source_dept_id: str
    target_dept_id: str
    patient_id:     str
    transfer_ts:    datetime
    acuity:         AcuityLevel

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


# ─── Context Events ───────────────────────────────────────────────────────────

class WeatherEvent(BaseModel):
    event_id:       str    = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:     str    = EventType.WEATHER
    hospital_id:    str
    timestamp:      datetime
    temperature_c:  float
    precipitation:  float  = Field(ge=0)    # mm/hr
    is_severe:      bool   = False
    weather_score:  float  = Field(ge=0, le=1)

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class FluIndexEvent(BaseModel):
    event_id:    str       = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:  str       = EventType.FLU_INDEX
    hospital_id: str
    timestamp:   datetime
    flu_index:   float     = Field(ge=0, le=10)
    trend:       str       = "stable"       # rising | stable | falling

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")


class TrafficEvent(BaseModel):
    event_id:        str   = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:      str   = EventType.TRAFFIC
    hospital_id:     str
    timestamp:       datetime
    congestion_score: float = Field(ge=0, le=1)
    avg_speed_kmh:   float  = Field(ge=0)

    def to_kafka_payload(self) -> bytes:
        return self.model_dump_json().encode("utf-8")
