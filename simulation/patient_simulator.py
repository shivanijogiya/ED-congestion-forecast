"""
Patient event simulator using Poisson arrival process with:
- Time-varying arrival rates (peaks 10am-2pm, 6pm-10pm)
- Acuity-conditioned LOS drawn from log-normal distribution
- Realistic surge events (flu season, mass casualty incidents)
"""
import random
import math
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Generator, Optional
from simulation.schemas import (
    PatientArrivalEvent, PatientDischargeEvent, PatientTransferEvent, AcuityLevel
)
from simulation.hospital_topology import Hospital, get_hospital_map

logger = logging.getLogger(__name__)

# ─── Arrival rate profiles (arrivals/hour by hour of day) ────────────────────
HOURLY_ARRIVAL_RATES = [
    2, 1.5, 1, 1, 1.2, 1.5,      # 0-5 AM  (overnight low)
    2.5, 4, 6, 8, 9, 10,          # 6-11 AM (morning surge)
    10, 9, 8, 7, 7, 8,            # 12-17 PM (midday → afternoon)
    9, 10, 9, 7, 5, 3,            # 18-23 PM (evening surge)
]

# LOS (minutes) parameters by acuity: (mean_log, std_log)
LOS_PARAMS = {
    AcuityLevel.CRITICAL:    (5.0, 0.3),   # ~150 min mean
    AcuityLevel.EMERGENT:    (4.6, 0.4),   # ~100 min mean
    AcuityLevel.URGENT:      (4.2, 0.5),   # ~67 min mean
    AcuityLevel.LESS_URGENT: (3.8, 0.5),   # ~45 min mean
    AcuityLevel.NON_URGENT:  (3.4, 0.6),   # ~30 min mean
}

# Acuity distribution (fraction) by department type
ACUITY_DIST = {
    "triage":     [0.05, 0.15, 0.35, 0.30, 0.15],
    "fast_track": [0.00, 0.02, 0.15, 0.45, 0.38],
    "resus":      [0.40, 0.50, 0.10, 0.00, 0.00],
    "obs":        [0.05, 0.25, 0.45, 0.20, 0.05],
    "boarding":   [0.10, 0.30, 0.40, 0.15, 0.05],
    "discharge":  [0.00, 0.05, 0.20, 0.40, 0.35],
    "radiology":  [0.02, 0.10, 0.30, 0.35, 0.23],
}

# Departments that receive direct arrivals (via ambulance/walk-in)
ENTRY_DEPT_TYPES = {"triage", "fast_track"}

# Transfer probability matrix: (source_type, target_type) → probability
TRANSFER_PROBS = {
    ("triage",    "resus"):     0.15,
    ("triage",    "fast_track"):0.25,
    ("triage",    "obs"):       0.30,
    ("resus",     "obs"):       0.40,
    ("resus",     "boarding"):  0.35,
    ("obs",       "boarding"):  0.30,
    ("obs",       "discharge"): 0.25,
    ("fast_track","discharge"): 0.50,
}


class PatientSimulator:
    def __init__(
        self,
        hospital: Hospital,
        surge_probability: float = 0.02,
        surge_multiplier: float = 2.5,
        flu_season: bool = False,
    ):
        self.hospital = hospital
        self.surge_probability = surge_probability
        self.surge_multiplier = surge_multiplier
        self.flu_multiplier = 1.4 if flu_season else 1.0
        self._active_patients: dict[str, dict] = {}  # patient_id → info
        self._dept_map = {d.dept_id: d for d in hospital.departments}

    def _arrival_rate(self, dt: datetime) -> float:
        """Poisson rate (arrivals/hour) at given time with optional surge."""
        base_rate = HOURLY_ARRIVAL_RATES[dt.hour]
        # Day-of-week: weekends ~20% higher
        if dt.weekday() >= 5:
            base_rate *= 1.2
        base_rate *= self.flu_multiplier
        # Random surge events (simulates mass casualty, events)
        if random.random() < self.surge_probability:
            base_rate *= self.surge_multiplier
        return base_rate

    def _sample_acuity(self, dept_type: str) -> AcuityLevel:
        dist = ACUITY_DIST.get(dept_type, ACUITY_DIST["triage"])
        r = random.random()
        cumulative = 0.0
        for level, prob in zip(AcuityLevel, dist):
            cumulative += prob
            if r < cumulative:
                return level
        return AcuityLevel.NON_URGENT

    def _sample_los(self, acuity: AcuityLevel) -> float:
        mu, sigma = LOS_PARAMS[acuity]
        return math.exp(random.gauss(mu, sigma))

    def _get_entry_depts(self):
        return [d for d in self.hospital.departments if d.dept_type in ENTRY_DEPT_TYPES]

    def generate_arrivals(self, window_start: datetime, window_end: datetime) -> List[PatientArrivalEvent]:
        """Generate arrivals for a time window using a Poisson process."""
        events = []
        duration_hours = (window_end - window_start).total_seconds() / 3600
        entry_depts = self._get_entry_depts()
        if not entry_depts:
            return events

        rate = self._arrival_rate(window_start) * duration_hours
        num_arrivals = max(0, int(random.gauss(rate, math.sqrt(rate))))

        for _ in range(num_arrivals):
            dept = random.choice(entry_depts)
            acuity = self._sample_acuity(dept.dept_type)
            los = self._sample_los(acuity)
            # Spread arrivals uniformly across the window
            offset = random.uniform(0, (window_end - window_start).total_seconds())
            arrival_ts = window_start + timedelta(seconds=offset)
            patient_id = str(uuid.uuid4())

            event = PatientArrivalEvent(
                hospital_id=self.hospital.hospital_id,
                dept_id=dept.dept_id,
                patient_id=patient_id,
                acuity=acuity,
                arrival_ts=arrival_ts,
                expected_los_minutes=los,
            )
            self._active_patients[patient_id] = {
                "dept_id": dept.dept_id,
                "dept_type": dept.dept_type,
                "acuity": acuity,
                "arrival_ts": arrival_ts,
                "expected_discharge_ts": arrival_ts + timedelta(minutes=los),
            }
            events.append(event)

        return events

    def generate_discharges(self, now: datetime) -> List[PatientDischargeEvent]:
        """Discharge patients whose expected LOS has elapsed."""
        events = []
        to_discharge = [
            pid for pid, info in self._active_patients.items()
            if info["expected_discharge_ts"] <= now
        ]
        for patient_id in to_discharge:
            info = self._active_patients.pop(patient_id)
            actual_los = (now - info["arrival_ts"]).total_seconds() / 60
            event = PatientDischargeEvent(
                hospital_id=self.hospital.hospital_id,
                dept_id=info["dept_id"],
                patient_id=patient_id,
                discharge_ts=now,
                actual_los_minutes=max(1.0, actual_los),
            )
            events.append(event)
        return events

    def generate_transfers(self, now: datetime) -> List[PatientTransferEvent]:
        """Probabilistically transfer some active patients between departments."""
        events = []
        edge_map = {
            (e.source, e.target): e
            for e in self.hospital.edges
            if e.edge_type == "TRANSFER"
        }
        dept_type_map = {d.dept_id: d.dept_type for d in self.hospital.departments}

        for patient_id, info in list(self._active_patients.items()):
            src_type = info["dept_type"]
            src_id = info["dept_id"]
            for (src, tgt), edge in edge_map.items():
                if src != src_id:
                    continue
                key = (src_type, dept_type_map.get(tgt, ""))
                prob = TRANSFER_PROBS.get(key, 0.0) * edge.base_weight * 0.1
                if random.random() < prob:
                    event = PatientTransferEvent(
                        hospital_id=self.hospital.hospital_id,
                        source_dept_id=src_id,
                        target_dept_id=tgt,
                        patient_id=patient_id,
                        transfer_ts=now,
                        acuity=info["acuity"],
                    )
                    # Update patient location
                    self._active_patients[patient_id]["dept_id"] = tgt
                    self._active_patients[patient_id]["dept_type"] = dept_type_map.get(tgt, src_type)
                    events.append(event)
                    break  # One transfer per patient per tick
        return events

    @property
    def active_patient_count(self) -> int:
        return len(self._active_patients)
