"""
Simulates external context signals correlated with patient arrivals:
- Weather: temperature, precipitation, severity (every 15 min)
- Flu Index: CDC-like weekly trend (daily)
- Traffic: road congestion per hospital (every 5 min)
"""
import math
import random
import logging
from datetime import datetime
from typing import List
from simulation.schemas import WeatherEvent, FluIndexEvent, TrafficEvent
from simulation.hospital_topology import HOSPITALS

logger = logging.getLogger(__name__)


class WeatherSimulator:
    """Generates weather events with realistic seasonal and diurnal patterns."""

    # Typical summer baseline for Gujarat, India
    BASE_TEMP_C = 32.0
    TEMP_AMPLITUDE = 8.0   # Diurnal swing

    def __init__(self, severe_probability: float = 0.03):
        self.severe_probability = severe_probability
        self._precipitation = 0.0

    def generate(self, now: datetime) -> List[WeatherEvent]:
        events = []
        # Diurnal temperature cycle (peak ~14:00)
        hour_rad = (now.hour - 14) * math.pi / 12
        temp = self.BASE_TEMP_C + self.TEMP_AMPLITUDE * math.cos(hour_rad)
        temp += random.gauss(0, 1.5)  # noise

        # Rain simulation: random onset
        if random.random() < 0.02:
            self._precipitation = random.uniform(2, 15)  # mm/hr
        elif self._precipitation > 0:
            self._precipitation *= 0.8  # exponential decay
        if self._precipitation < 0.1:
            self._precipitation = 0.0

        is_severe = (
            self._precipitation > 10
            or random.random() < self.severe_probability
        )
        # Weather score: 0=fine, 1=extreme
        weather_score = min(1.0, (self._precipitation / 15) + (0.3 if is_severe else 0))

        for hospital in HOSPITALS:
            events.append(WeatherEvent(
                hospital_id=hospital.hospital_id,
                timestamp=now,
                temperature_c=round(temp + random.gauss(0, 0.5), 1),
                precipitation=round(max(0, self._precipitation + random.gauss(0, 0.5)), 2),
                is_severe=is_severe,
                weather_score=round(min(1.0, max(0.0, weather_score + random.gauss(0, 0.05))), 3),
            ))
        return events


class FluIndexSimulator:
    """
    Simulates flu index using a seasonal sine wave with random shocks.
    Peaks in winter; low in summer.
    """

    def __init__(self, base_index: float = 2.0):
        self.base_index = base_index
        self._current_index = base_index
        self._shock_active = False
        self._shock_remaining = 0

    def generate(self, now: datetime) -> List[FluIndexEvent]:
        # Seasonal component: peak in Jan (day 15)
        day_of_year = now.timetuple().tm_yday
        seasonal = 3.0 * math.sin((day_of_year - 75) * math.pi / 182)

        # Random flu spike
        if not self._shock_active and random.random() < 0.005:
            self._shock_active = True
            self._shock_remaining = random.randint(7, 21)  # days
        if self._shock_active:
            shock = random.uniform(2.0, 4.0)
            self._shock_remaining -= 1
            if self._shock_remaining <= 0:
                self._shock_active = False
        else:
            shock = 0.0

        raw = self.base_index + seasonal + shock + random.gauss(0, 0.2)
        self._current_index = round(max(0.0, min(10.0, raw)), 2)

        trend = "stable"
        if shock > 0:
            trend = "rising"
        elif seasonal < -2:
            trend = "falling"

        events = []
        for hospital in HOSPITALS:
            events.append(FluIndexEvent(
                hospital_id=hospital.hospital_id,
                timestamp=now,
                flu_index=self._current_index + round(random.gauss(0, 0.1), 2),
                trend=trend,
            ))
        return events


class TrafficSimulator:
    """
    Simulates road congestion with rush-hour peaks and random incidents.
    """

    RUSH_HOURS = {(7, 9), (17, 19)}

    def __init__(self):
        self._incidents: dict[str, float] = {}   # hospital_id → incident congestion

    def _is_rush_hour(self, hour: int) -> bool:
        return any(start <= hour < end for start, end in self.RUSH_HOURS)

    def generate(self, now: datetime) -> List[TrafficEvent]:
        events = []
        is_rush = self._is_rush_hour(now.hour)
        base_congestion = 0.6 if is_rush else 0.25

        for hospital in HOSPITALS:
            hid = hospital.hospital_id

            # Random traffic incidents
            if random.random() < 0.01:
                self._incidents[hid] = random.uniform(0.4, 0.9)
            if hid in self._incidents:
                self._incidents[hid] *= 0.9
                if self._incidents[hid] < 0.05:
                    del self._incidents[hid]

            incident_boost = self._incidents.get(hid, 0.0)
            congestion = min(1.0, base_congestion + incident_boost + random.gauss(0, 0.05))
            speed = max(5.0, 80.0 * (1 - congestion) + random.gauss(0, 3))

            events.append(TrafficEvent(
                hospital_id=hid,
                timestamp=now,
                congestion_score=round(max(0.0, min(1.0, congestion)), 3),
                avg_speed_kmh=round(max(0.0, speed), 1),
            ))
        return events
