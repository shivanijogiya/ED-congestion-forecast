"""
APScheduler-based inference scheduler.
Runs predictions every 5 minutes for all hospitals,
writes results to Cassandra and Elasticsearch.
"""
import logging
import math
import random
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from simulation.hospital_topology import HOSPITALS
from graph_model.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

PREDICTION_INTERVAL_MINUTES = 5


def _generate_demo_features(hospital, sequence_len: int = 24):
    """Generate synthetic feature sequence when Cassandra is unavailable."""
    base_occ = random.uniform(0.3, 0.85)
    feature_seq = []
    for t in range(sequence_len):
        dept_features = {}
        for dept in hospital.departments:
            hour = t % 24
            features = [
                min(1.0, base_occ + random.gauss(0, 0.05)),
                random.uniform(2, 10),
                random.uniform(0.05, 0.3),
                random.uniform(10, 60),
                random.gauss(0, 0.2),
                random.uniform(0, 0.5),
                random.uniform(1, 7),
                random.uniform(0.1, 0.6),
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
            ]
            dept_features[dept.dept_id] = features
        feature_seq.append(dept_features)
    return feature_seq


def run_inference_job(predictor, es_writer=None, cassandra_writer=None):
    """Single inference job: predict for all hospitals, write results."""
    logger.info(f"Running inference at {datetime.utcnow().isoformat()}")
    for hospital in HOSPITALS:
        try:
            feature_seq = _generate_demo_features(hospital, sequence_len=predictor.config.sequence_len)
            result = predictor.predict(hospital.hospital_id, feature_seq)
            logger.info(
                f"{hospital.hospital_id}: severity={result['hospital_severity']}, "
                f"max_congestion={result['hospital_max_congestion']:.2f}"
            )
            if es_writer:
                es_writer.index_prediction(result)
            if cassandra_writer:
                cassandra_writer.write_prediction(result)

        except Exception as e:
            logger.error(f"Inference failed for {hospital.hospital_id}: {e}", exc_info=True)


def start_scheduler(predictor, es_writer=None, cassandra_writer=None):
    scheduler = BlockingScheduler()
    scheduler.add_job(
        func=run_inference_job,
        args=[predictor, es_writer, cassandra_writer],
        trigger=IntervalTrigger(minutes=PREDICTION_INTERVAL_MINUTES),
        id="ed_inference",
        name="ED Congestion Forecast",
        max_instances=1,
        replace_existing=True,
    )
    logger.info(f"Inference scheduler started (every {PREDICTION_INTERVAL_MINUTES} minutes)")
    scheduler.start()
