.PHONY: up down logs install train simulate api test clean

# ─── Infrastructure ───────────────────────────────────────────────────────────
up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 30
	@echo "All services started. Visit:"
	@echo "  API:     http://localhost:8000/docs"
	@echo "  Kibana:  http://localhost:5601"
	@echo "  Spark:   http://localhost:8080"

down:
	docker-compose down -v

logs:
	docker-compose logs -f

restart:
	docker-compose restart

# ─── Python setup ─────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-torch-geo:
	pip install torch-scatter torch-sparse torch-geometric \
	  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# ─── Model training ───────────────────────────────────────────────────────────
train:
	python -m graph_model.training.train_pipeline --config config/model_config.yaml

train-demo:
	python -m graph_model.training.train_pipeline

# ─── Simulation ───────────────────────────────────────────────────────────────
simulate:
	python -m simulation.run_simulation --realtime --speedup 60

simulate-flu:
	python -m simulation.run_simulation --realtime --speedup 60 --flu-season

# ─── API server ───────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─── Spark streaming ──────────────────────────────────────────────────────────
spark-stream:
	python -c "
from spark_processing.spark_session_factory import create_spark_session
from spark_processing.streaming.patient_stream_processor import run_patient_stream
from spark_processing.streaming.context_stream_processor import run_context_stream
spark = create_spark_session()
q1 = run_patient_stream(spark)
q2 = run_context_stream(spark)
spark.streams.awaitAnyTermination()
"

# ─── Inference scheduler ──────────────────────────────────────────────────────
infer:
	python -c "
from graph_model.inference.inference_scheduler import start_scheduler, run_inference_job
# Demo mode without a real checkpoint
class DemoPredictor:
    class config:
        sequence_len = 24
    def predict(self, h, seq):
        return {'hospital_id': h, 'departments': [], 'hospital_max_congestion': 0.5, 'hospital_severity': 'green'}
run_inference_job(DemoPredictor())
"

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf checkpoints/ runs/ /tmp/ed_checkpoint

fmt:
	black . && isort .
