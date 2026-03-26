-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS ed_forecasting
WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 1
};

USE ed_forecasting;

-- Feature windows: per-department feature vectors indexed by time
CREATE TABLE IF NOT EXISTS feature_windows (
    hospital_id     TEXT,
    dept_id         TEXT,
    window_ts       TIMESTAMP,
    feature_vector  LIST<DOUBLE>,
    occupancy_ratio DOUBLE,
    arrival_rate    DOUBLE,
    severity_index  DOUBLE,
    avg_wait_time   DOUBLE,
    los_deviation   DOUBLE,
    weather_score   DOUBLE,
    flu_index       DOUBLE,
    traffic_score   DOUBLE,
    hour_sin        DOUBLE,
    hour_cos        DOUBLE,
    PRIMARY KEY ((hospital_id, dept_id), window_ts)
) WITH CLUSTERING ORDER BY (window_ts DESC)
  AND default_time_to_live = 604800;  -- 7 days TTL

-- Predictions: per-department congestion forecasts
CREATE TABLE IF NOT EXISTS predictions (
    hospital_id         TEXT,
    dept_id             TEXT,
    prediction_ts       TIMESTAMP,
    horizon_1h          DOUBLE,
    horizon_2h          DOUBLE,
    horizon_4h          DOUBLE,
    horizon_8h          DOUBLE,
    confidence          DOUBLE,
    severity_label      TEXT,
    model_version       TEXT,
    PRIMARY KEY ((hospital_id, dept_id), prediction_ts)
) WITH CLUSTERING ORDER BY (prediction_ts DESC)
  AND default_time_to_live = 2592000;  -- 30 days TTL

-- Model metadata: version tracking
CREATE TABLE IF NOT EXISTS model_metadata (
    model_version   TEXT PRIMARY KEY,
    trained_at      TIMESTAMP,
    config_json     TEXT,
    metrics_json    TEXT,
    is_active       BOOLEAN
);

-- Department registry: static graph topology
CREATE TABLE IF NOT EXISTS department_registry (
    hospital_id     TEXT,
    dept_id         TEXT,
    dept_name       TEXT,
    dept_type       TEXT,
    capacity        INT,
    floor_number    INT,
    lat             DOUBLE,
    lon             DOUBLE,
    PRIMARY KEY (hospital_id, dept_id)
);

-- Graph edges: department relationships
CREATE TABLE IF NOT EXISTS graph_edges (
    hospital_id     TEXT,
    source_dept     TEXT,
    target_dept     TEXT,
    edge_type       TEXT,
    base_weight     DOUBLE,
    PRIMARY KEY ((hospital_id, source_dept), target_dept)
);

-- Alerts: threshold breach events
CREATE TABLE IF NOT EXISTS alerts (
    hospital_id     TEXT,
    dept_id         TEXT,
    alert_ts        TIMESTAMP,
    alert_type      TEXT,
    severity        TEXT,
    threshold       DOUBLE,
    actual_value    DOUBLE,
    predicted_value DOUBLE,
    resolved        BOOLEAN,
    PRIMARY KEY ((hospital_id, dept_id), alert_ts)
) WITH CLUSTERING ORDER BY (alert_ts DESC)
  AND default_time_to_live = 7776000;  -- 90 days TTL
