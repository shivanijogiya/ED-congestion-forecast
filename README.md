# Hospital ED Congestion Forecasting

> **Graph-Aware Contextual Deep Learning for Real-Time Emergency Department Congestion Forecasting using Streaming Big Data**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Kafka-7.5-231F20?logo=apachekafka&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Spark-3.5-E25A1C?logo=apachespark&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A **production-grade streaming Big Data system** that predicts Emergency Department congestion **1–8 hours ahead** using a hybrid deep learning model combining Graph Attention Networks, LSTM, and Multi-head Attention — fed by real-time hospital events and contextual signals.

---

##  Problem Statement

Traditional ED forecasting models have **three critical limitations**:

| Limitation | Impact |
|-----------|--------|
| Ignore inter-department relationships | Surge in Triage is not linked to overflow in Resus |
| Ignore external context signals | Flu trends, weather, traffic all drive ED demand |
| Static batch-trained models | Cannot adapt to real-time conditions |

**This system solves all three.**

---

##  Quick Start (No Docker Needed — 60 seconds)

### Mac / Linux
```bash
git clone https://github.com/your-username/ed-congestion-forecast.git
cd ed-congestion-forecast
chmod +x start.sh && ./start.sh
```

### Windows
```
Double-click  start.bat
```

### Manual
```bash
pip install fastapi uvicorn pydantic pydantic-settings httpx pytest pytest-asyncio numpy scipy
python -m pytest tests/ -v
python main.py demo
```

Then open **http://localhost:8000** in your browser.

---

##  Dashboard

The system ships with a full real-time web dashboard at `http://localhost:8000`:

- **Hospital Overview** — 6 hospital cards with live severity indicators (🟢 green / 🟡 amber / 🔴 red)
- **Department Forecast** — Bar chart of all 7 departments × 4 horizons (1h / 2h / 4h / 8h)
- **Context Signals** — Weather score, Flu index, Traffic congestion per hospital
- **24-Hour History** — Line chart of occupancy trends per department
- **Auto-refresh** — Updates every 30 seconds

**API Docs (Swagger UI):** `http://localhost:8000/docs`

---

##  System Architecture

```
Hospital Events          External Signals
(arrivals/discharges     (weather / flu index /
 transfers)               traffic congestion)
        │                        │
        ▼                        ▼
   Kafka Producers ──────────────┘
        │
   Apache Kafka
   (6 topics, partitioned by hospital_id)
        │
   Spark Structured Streaming
   (stateful occupancy + watermark-bounded stream join)
        │
   Feature Engineering
   (10-dim vector per department per 10-min window)
        │
   ┌────────────────────────────────────────────┐
   │       Graph-LSTM-Attention Model           │
   │                                            │
   │  GATv2Conv ──► LSTM ──► Multi-head Attn   │
   │  (spatial)    (temporal)  (historical ctx) │
   │                                            │
   │  Output: congestion score per dept         │
   │          horizons: 1h / 2h / 4h / 8h      │
   └────────────────────────────────────────────┘
        │
   Cassandra (operational) + Hive (archival)
        │
   Elasticsearch + Kibana dashboard
        │
   FastAPI REST API + Web UI
```

---

##  Model Architecture

### Why this combination?

| Component | Choice | Reason |
|-----------|--------|--------|
| GNN | **GATv2Conv** | Dynamic attention — captures asymmetric patient flow (Triage→Resus ≠ Resus→Triage) |
| Temporal | **LSTM** (weight-tied across nodes) | Temporal autocorrelation; weight-sharing reduces overfitting on small departments |
| Decoder | **Multi-head Self-Attention** | Selects which historical moment (e.g., a surge 3h ago) matters most for prediction |
| Loss | **Asymmetric MSE + Hinge** | Underpredicting high congestion penalized 3× more — clinically safer |

### Input Feature Vector (10 dimensions per department per timestep)

| # | Feature | Source |
|---|---------|--------|
| 0 | Occupancy ratio (census / capacity) | Patient events |
| 1 | Arrival rate (arrivals / window) | Patient events |
| 2 | Severity index (ESI 1–2 fraction) | Patient events |
| 3 | Average wait time (minutes) | Patient events |
| 4 | LOS deviation from baseline | Patient events |
| 5 | Weather severity score | External / simulator |
| 6 | Flu trend index (0–10) | External / simulator |
| 7 | Traffic congestion score | External / simulator |
| 8 | Hour-of-day sine encoding | Computed |
| 9 | Hour-of-day cosine encoding | Computed |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Event Streaming | Apache Kafka (Confluent) |
| Stream Processing | Apache Spark Structured Streaming |
| Graph Neural Network | PyTorch Geometric — GATv2Conv |
| Temporal Model | PyTorch LSTM (weight-tied) |
| Attention | PyTorch Multi-head Self-Attention |
| Operational DB | Apache Cassandra |
| Analytics Store | Apache Hive |
| Search & Indexing | Elasticsearch |
| Dashboard | Kibana + Custom Web UI |
| REST API | FastAPI |
| Infrastructure | Docker Compose |

---

## Project Structure

```
ed-congestion-forecast/
│
├── 📂 simulation/               # Data generators
│   ├── hospital_topology.py     # 6 hospitals, 7 depts each, graph edges
│   ├── patient_simulator.py     # Poisson arrivals, LOS, surge events
│   ├── external_context_simulator.py  # Weather, flu, traffic
│   ├── schemas.py               # Pydantic event models
│   └── run_simulation.py        # Orchestrates all simulators → Kafka
│
├── 📂 kafka_layer/              # Kafka producers & consumers
│   └── producers/
│       ├── patient_event_producer.py
│       └── context_event_producer.py
│
├── 📂 spark_processing/         # Spark Structured Streaming
│   └── streaming/
│       ├── patient_stream_processor.py   # Stateful occupancy features
│       └── context_stream_processor.py   # Context signal aggregation
│
├── 📂 graph_model/              # The ML model
│   ├── layers/
│   │   ├── gnn_encoder.py       # GATv2Conv — spatial encoding
│   │   ├── lstm_temporal.py     # LSTM — temporal encoding
│   │   └── attention_decoder.py # Multi-head attention — decoding
│   ├── model/
│   │   ├── ed_forecast_model.py # Top-level nn.Module
│   │   ├── model_config.py      # Hyperparameter dataclass
│   │   └── loss_functions.py    # Asymmetric MSE + hinge loss
│   ├── training/
│   │   ├── dataset.py           # PyTorch Dataset (demo + Cassandra)
│   │   ├── trainer.py           # Training loop + early stopping
│   │   └── train_pipeline.py    # Entry point for training
│   └── inference/
│       ├── predictor.py         # Load checkpoint → predictions
│       └── inference_scheduler.py  # APScheduler every 5 min
│
├── 📂 storage/                  # Database writers
│   ├── cassandra_writer.py
│   └── elasticsearch_writer.py
│
├── 📂 api/                      # FastAPI server
│   └── main.py                  # All endpoints + static file serving
│
├── 📂 static/                   # Web dashboard
│   ├── index.html               # Dashboard layout
│   ├── style.css                # Dark theme UI
│   └── app.js                   # Charts, API calls, auto-refresh
│
├── 📂 monitoring/
│   └── model_drift_detector.py  # PSI-based drift → auto-retrain trigger
│
├── 📂 infra/
│   ├── kafka/create_topics.sh   # Kafka topic setup
│   ├── cassandra/init_keyspace.cql  # DB schema
│   └── elasticsearch/index_mappings.json
│
├── 📂 tests/
│   ├── unit/test_simulation.py  # 8 simulation tests
│   ├── unit/test_model_layers.py  # GNN/LSTM/Attention shape tests
│   ├── unit/test_graph_construction.py
│   └── integration/test_api.py   # 13 API endpoint tests
│
├── 📂 config/
│   ├── model_config.yaml        # Model hyperparameters
│   └── app_config.yaml          # Runtime configuration
│
├── docker-compose.yml           # Full infrastructure
├── Dockerfile                   # API container
├── requirements.txt             # All Python dependencies
├── pyproject.toml               # Package config
├── Makefile                     # Dev shortcuts
├── start.sh                     #  One-command Mac/Linux startup
├── start.bat                    # 🚀 One-command Windows startup
└── main.py                      # Master entry point
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/forecast/{hospital_id}` | Congestion predictions (1h–8h) |
| `GET` | `/hospitals` | Graph topology (departments + edges) |
| `GET` | `/history/{hospital_id}?hours=24` | Historical occupancy |
| `GET` | `/health` | Infrastructure + model health |
| `GET` | `/docs` | Swagger UI |

### Example: `/forecast/H1`
```json
{
  "hospital_id": "H1",
  "hospital_name": "City General Hospital",
  "hospital_severity": "amber",
  "hospital_max_congestion": 0.79,
  "departments": [
    {
      "dept_name": "Triage",
      "dept_type": "triage",
      "current_occupancy": 0.72,
      "severity_label": "amber",
      "forecasts": { "1h": 0.74, "2h": 0.79, "4h": 0.81, "8h": 0.68 },
      "contributing_factors": {
        "weather_score": 0.31,
        "flu_index": 5.2,
        "traffic_score": 0.58
      }
    }
  ],
  "model_info": {
    "architecture": "Graph-LSTM-Attention (GATv2 + LSTM + MultiheadAttention)",
    "version": "1.0.0"
  }
}
```

---

## 🐳 Full Infrastructure (Docker)

```bash
# Start everything (Kafka, Spark, Cassandra, Elasticsearch, Kibana)
docker-compose up -d

# Stream live patient + context events into Kafka
python -m simulation.run_simulation --realtime --speedup 60

# Start Spark Streaming feature engineering
python main.py spark

# Train the model on synthetic data
python main.py train

# Start inference scheduler (predictions every 5 min)
python main.py infer
```

**Services available:**
| Service | URL |
|---------|-----|
| Web Dashboard | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Kibana | http://localhost:5601 |
| Spark UI | http://localhost:8080 |

---

## ☁️ Deploy to Cloud

### Deploy to Railway (one click)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Deploy to Render
1. Push repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Deploy with Docker
```bash
docker build -t ed-forecast .
docker run -p 8000:8000 ed-forecast
```

---

## ✅ Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (no infra needed)
python -m pytest tests/unit/ -v

# API integration tests
python -m pytest tests/integration/ -v
```

Expected output: **21 passed** in < 1 second.

---

## 🔬 Research Contributions

1. **Graph-Aware Forecasting** — Hospital departments modeled as a directed graph; GATv2 learns asymmetric inter-department attention weights
2. **Context-Aware Features** — Weather, flu trends, and traffic signals fused with patient-flow features
3. **Clinically-Motivated Loss** — Asymmetric penalty: underpredicting high congestion is 3× worse than overprediction
4. **Streaming Architecture** — End-to-end real-time pipeline from raw events to predictions in < 60 seconds
5. **Automatic Drift Detection** — PSI-based model monitoring triggers retraining when distribution shifts

---

## 📄 License

MIT © 2026

---

##  Author

Built as a Big Data course project demonstrating research-grade streaming ML systems.
