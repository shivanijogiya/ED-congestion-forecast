"""
Writes predictions and feature windows to Cassandra.
"""
import logging
import os
from datetime import datetime
from typing import Dict
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

logger = logging.getLogger(__name__)

CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "localhost")
KEYSPACE = "ed_forecasting"


class CassandraWriter:
    def __init__(self):
        self.cluster = Cluster(
            [CASSANDRA_HOST],
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        )
        self.session = self.cluster.connect(KEYSPACE)
        self._prep_statements()

    def _prep_statements(self):
        self._insert_prediction = self.session.prepare("""
            INSERT INTO predictions
            (hospital_id, dept_id, prediction_ts, horizon_1h, horizon_2h,
             horizon_4h, horizon_8h, confidence, severity_label, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)

    def write_prediction(self, prediction_result: Dict) -> None:
        ts = datetime.utcnow()
        hospital_id = prediction_result["hospital_id"]
        model_version = prediction_result.get("model_version", "1.0.0")

        for dept in prediction_result.get("departments", []):
            forecasts = dept["forecasts"]
            self.session.execute(self._insert_prediction, (
                hospital_id,
                dept["dept_id"],
                ts,
                forecasts.get("1h", 0.0),
                forecasts.get("2h", 0.0),
                forecasts.get("4h", 0.0),
                forecasts.get("8h", 0.0),
                0.85,   # placeholder confidence
                dept["severity_label"],
                model_version,
            ))

    def close(self):
        self.cluster.shutdown()
