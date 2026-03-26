"""
Indexes prediction events and alerts into Elasticsearch for Kibana visualization.
Uses bulk indexing with buffered writes.
"""
import logging
import os
from datetime import datetime
from typing import List, Dict
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)

ES_HOST   = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
INDEX_PREFIX = "ed-predictions"


class ElasticsearchWriter:
    def __init__(self):
        self.client = Elasticsearch(
            ES_HOST,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )
        self._buffer: List[Dict] = []
        self._buffer_size = 100
        self._ensure_index()

    def _ensure_index(self):
        index = f"{INDEX_PREFIX}"
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body={
                "mappings": {
                    "properties": {
                        "hospital_id":          {"type": "keyword"},
                        "dept_id":              {"type": "keyword"},
                        "dept_name":            {"type": "keyword"},
                        "dept_type":            {"type": "keyword"},
                        "prediction_timestamp": {"type": "date"},
                        "horizon_1h":           {"type": "float"},
                        "horizon_2h":           {"type": "float"},
                        "horizon_4h":           {"type": "float"},
                        "horizon_8h":           {"type": "float"},
                        "max_congestion":       {"type": "float"},
                        "severity_label":       {"type": "keyword"},
                        "model_version":        {"type": "keyword"},
                    }
                }
            })
            logger.info(f"Created index: {index}")

    def index_prediction(self, prediction_result: Dict) -> None:
        """Buffer a prediction result; flush when buffer is full."""
        ts = prediction_result.get("prediction_timestamp", datetime.utcnow().isoformat())
        hospital_id = prediction_result["hospital_id"]

        for dept in prediction_result.get("departments", []):
            doc = {
                "_index": INDEX_PREFIX,
                "_source": {
                    "hospital_id":          hospital_id,
                    "dept_id":              dept["dept_id"],
                    "dept_name":            dept["dept_name"],
                    "dept_type":            dept["dept_type"],
                    "prediction_timestamp": ts,
                    "horizon_1h":           dept["forecasts"].get("1h", 0),
                    "horizon_2h":           dept["forecasts"].get("2h", 0),
                    "horizon_4h":           dept["forecasts"].get("4h", 0),
                    "horizon_8h":           dept["forecasts"].get("8h", 0),
                    "max_congestion":       dept["max_congestion"],
                    "severity_label":       dept["severity_label"],
                    "model_version":        prediction_result.get("model_version", "1.0.0"),
                }
            }
            self._buffer.append(doc)

        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def index_alert(self, alert: Dict) -> None:
        self.client.index(index="ed-alerts", document={
            **alert,
            "indexed_at": datetime.utcnow().isoformat(),
        })

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            success, errors = helpers.bulk(self.client, self._buffer, raise_on_error=False)
            if errors:
                logger.error(f"ES bulk errors: {errors[:3]}")
            logger.debug(f"Flushed {success} docs to Elasticsearch")
        except Exception as e:
            logger.error(f"ES bulk write failed: {e}")
        finally:
            self._buffer.clear()

    def flush(self) -> None:
        self._flush()

    def close(self) -> None:
        self._flush()
        self.client.close()
