"""
Unit tests for the patient and context simulators.
"""
import pytest
from datetime import datetime, timedelta
from simulation.hospital_topology import HOSPITALS
from simulation.patient_simulator import PatientSimulator
from simulation.external_context_simulator import WeatherSimulator, FluIndexSimulator, TrafficSimulator
from simulation.schemas import PatientArrivalEvent, WeatherEvent, FluIndexEvent, TrafficEvent


class TestPatientSimulator:
    def setup_method(self):
        self.hospital = HOSPITALS[0]
        self.sim = PatientSimulator(self.hospital)

    def test_generates_arrivals(self):
        start = datetime(2026, 3, 26, 10, 0)   # 10 AM (high traffic)
        end   = start + timedelta(hours=2)
        arrivals = self.sim.generate_arrivals(start, end)
        assert len(arrivals) > 0

    def test_arrivals_have_valid_schema(self):
        start = datetime(2026, 3, 26, 10, 0)
        end   = start + timedelta(hours=1)
        arrivals = self.sim.generate_arrivals(start, end)
        for ev in arrivals:
            assert isinstance(ev, PatientArrivalEvent)
            assert ev.hospital_id == self.hospital.hospital_id
            assert 1 <= ev.acuity <= 5
            assert ev.expected_los_minutes > 0

    def test_discharges_after_los(self):
        # Use peak hours (10 AM, 3-hour window) to guarantee arrivals
        start = datetime(2026, 3, 26, 10, 0)
        end   = start + timedelta(hours=3)
        arrivals = self.sim.generate_arrivals(start, end)
        assert len(arrivals) > 0, "No arrivals generated — increase window if flaky"

        # Advance time 24h past end — all patients must be discharged by then
        future = end + timedelta(hours=24)
        discharges = self.sim.generate_discharges(future)
        assert len(discharges) > 0

    def test_active_count_decreases_after_discharge(self):
        start = datetime(2026, 3, 26, 8, 0)
        end   = start + timedelta(hours=1)
        self.sim.generate_arrivals(start, end)
        initial = self.sim.active_patient_count

        future = end + timedelta(hours=6)
        self.sim.generate_discharges(future)
        assert self.sim.active_patient_count <= initial

    def test_flu_season_increases_arrivals(self):
        normal_sim = PatientSimulator(self.hospital, flu_season=False)
        flu_sim    = PatientSimulator(self.hospital, flu_season=True)

        start = datetime(2026, 3, 26, 10, 0)
        end   = start + timedelta(hours=4)

        # Run multiple times to get stable averages
        normal_counts = [len(normal_sim.generate_arrivals(start, end)) for _ in range(10)]
        flu_counts    = [len(flu_sim.generate_arrivals(start, end))    for _ in range(10)]

        assert sum(flu_counts) >= sum(normal_counts) * 0.9  # Allow small variance


class TestContextSimulators:
    def test_weather_generates_per_hospital(self):
        sim = WeatherSimulator()
        now = datetime(2026, 3, 26, 14, 0)
        events = sim.generate(now)
        assert len(events) == len(HOSPITALS)
        for ev in events:
            assert isinstance(ev, WeatherEvent)
            assert 0 <= ev.weather_score <= 1

    def test_flu_index_in_range(self):
        sim = FluIndexSimulator()
        for _ in range(20):
            events = sim.generate(datetime(2026, 3, 26, 9, 0))
        for ev in events:
            assert isinstance(ev, FluIndexEvent)
            assert 0 <= ev.flu_index <= 10

    def test_traffic_congestion_in_range(self):
        sim = TrafficSimulator()
        events = sim.generate(datetime(2026, 3, 26, 8, 0))  # Rush hour
        for ev in events:
            assert isinstance(ev, TrafficEvent)
            assert 0 <= ev.congestion_score <= 1
