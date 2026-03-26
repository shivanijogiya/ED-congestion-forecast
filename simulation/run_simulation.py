"""
Main simulation entry point.
Runs all simulators and publishes events to Kafka in real-time.

Usage:
    python -m simulation.run_simulation [--realtime] [--speedup 60] [--flu-season]
"""
import argparse
import logging
import time
from datetime import datetime, timedelta

from simulation.hospital_topology import HOSPITALS
from simulation.patient_simulator import PatientSimulator
from simulation.external_context_simulator import (
    WeatherSimulator, FluIndexSimulator, TrafficSimulator
)
from kafka_layer.producers.patient_event_producer import PatientEventProducer
from kafka_layer.producers.context_event_producer import ContextEventProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TICK_INTERVAL_MINUTES = 10   # Simulate in 10-minute increments


def run(realtime: bool = False, speedup: int = 60, flu_season: bool = False):
    logger.info("Initializing simulators...")

    # One patient simulator per hospital
    patient_sims = {
        h.hospital_id: PatientSimulator(h, flu_season=flu_season)
        for h in HOSPITALS
    }
    weather_sim = WeatherSimulator()
    flu_sim     = FluIndexSimulator()
    traffic_sim = TrafficSimulator()

    patient_producer = PatientEventProducer()
    context_producer = ContextEventProducer()

    sim_time = datetime.now().replace(second=0, microsecond=0)
    tick_delta = timedelta(minutes=TICK_INTERVAL_MINUTES)

    weather_tick = 0    # publish every 15 min → every 1-2 ticks
    flu_tick = 0        # publish every 1440 min → every 144 ticks
    traffic_tick = 0    # publish every 5 min → every tick

    logger.info(f"Starting simulation from {sim_time} (speedup={speedup}x)")

    try:
        while True:
            window_start = sim_time
            window_end   = sim_time + tick_delta

            # ── Patient events ────────────────────────────────────────────
            for hospital_id, sim in patient_sims.items():
                arrivals  = sim.generate_arrivals(window_start, window_end)
                discharges = sim.generate_discharges(window_end)
                transfers  = sim.generate_transfers(window_end)

                for ev in arrivals:
                    patient_producer.publish_arrival(ev)
                for ev in discharges:
                    patient_producer.publish_discharge(ev)
                for ev in transfers:
                    patient_producer.publish_transfer(ev)

                if arrivals or discharges:
                    logger.info(
                        f"{hospital_id} | +{len(arrivals)} arrivals, "
                        f"-{len(discharges)} discharges, "
                        f"{sim.active_patient_count} active"
                    )

            # ── Traffic (every tick / 5-10 min) ───────────────────────────
            traffic_events = traffic_sim.generate(window_end)
            for ev in traffic_events:
                context_producer.publish_traffic(ev)

            # ── Weather (every ~15 min → every other tick) ────────────────
            weather_tick += 1
            if weather_tick >= 2:
                weather_events = weather_sim.generate(window_end)
                for ev in weather_events:
                    context_producer.publish_weather(ev)
                weather_tick = 0

            # ── Flu index (daily → every 144 ticks at 10min intervals) ────
            flu_tick += 1
            if flu_tick >= 144:
                flu_events = flu_sim.generate(window_end)
                for ev in flu_events:
                    context_producer.publish_flu_index(ev)
                flu_tick = 0

            patient_producer.flush()
            context_producer.flush()

            sim_time = window_end

            if realtime:
                sleep_secs = (TICK_INTERVAL_MINUTES * 60) / speedup
                time.sleep(sleep_secs)

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        patient_producer.close()
        context_producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ED Event Simulator")
    parser.add_argument("--realtime",   action="store_true", help="Sleep between ticks")
    parser.add_argument("--speedup",    type=int, default=60, help="Speedup factor for realtime mode")
    parser.add_argument("--flu-season", action="store_true", help="Enable flu season multiplier")
    args = parser.parse_args()

    run(realtime=args.realtime, speedup=args.speedup, flu_season=args.flu_season)
