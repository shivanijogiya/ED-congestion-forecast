"""
ED Congestion Forecasting System — Master Entry Point

Commands:
  python main.py api        — Start FastAPI server
  python main.py simulate   — Start patient + context event simulation
  python main.py train      — Train the Graph-LSTM-Attention model
  python main.py infer      — Start real-time inference scheduler
  python main.py spark      — Start Spark Streaming processors
  python main.py demo       — Run full demo (API + synthetic predictions)
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_api():
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


def cmd_simulate():
    from simulation.run_simulation import run
    run(realtime=True, speedup=60, flu_season=False)


def cmd_train():
    from graph_model.training.train_pipeline import run
    checkpoint = run()
    logger.info(f"Training complete. Checkpoint: {checkpoint}")


def cmd_infer():
    """Demo inference without a real checkpoint."""
    import math
    import random
    from simulation.hospital_topology import HOSPITALS
    from graph_model.model.ed_forecast_model import EDForecastModel
    from graph_model.model.model_config import ModelConfig
    from graph_model.graph_construction.hospital_graph_builder import HospitalGraphBuilder
    from graph_model.inference.predictor import get_severity

    config = ModelConfig()
    model  = EDForecastModel(config)
    model.eval()

    logger.info("Running demo inference (untrained model — random predictions expected)...")
    for hospital in HOSPITALS[:2]:
        builder = HospitalGraphBuilder(hospital)
        T = config.sequence_len
        feature_seq = [
            {d.dept_id: [random.gauss(0, 1)] * 10 for d in hospital.departments}
            for _ in range(T)
        ]
        import torch
        graph_seq = builder.build_graph_sequence(feature_seq)
        with torch.no_grad():
            preds = model([graph_seq], builder.num_nodes)
        logger.info(f"{hospital.hospital_id}: predictions shape={preds.shape}")
        for i, dept in enumerate(hospital.departments):
            p = preds[0, i].tolist()
            logger.info(f"  {dept.dept_name}: 1h={p[0]:.3f}, 2h={p[1]:.3f}, 4h={p[2]:.3f}, 8h={p[3]:.3f}")


def cmd_spark():
    from spark_processing.spark_session_factory import create_spark_session
    from spark_processing.streaming.patient_stream_processor import run_patient_stream
    from spark_processing.streaming.context_stream_processor import run_context_stream

    spark = create_spark_session()
    logger.info("Starting Spark Streaming...")
    q1 = run_patient_stream(spark)
    q2 = run_context_stream(spark)
    spark.streams.awaitAnyTermination()


def cmd_demo():
    """Start the API and print key demo info."""
    logger.info("=" * 60)
    logger.info("ED CONGESTION FORECASTING DEMO")
    logger.info("=" * 60)
    logger.info("Architecture: GATv2 → LSTM → Multi-head Attention")
    logger.info("Hospitals: 6  |  Departments per hospital: 7")
    logger.info("Forecast horizons: 1h, 2h, 4h, 8h")
    logger.info("")
    logger.info("Starting API server on http://localhost:8000")
    logger.info("API docs:  http://localhost:8000/docs")
    logger.info("Forecast:  http://localhost:8000/forecast/H1")
    logger.info("Topology:  http://localhost:8000/hospitals")
    logger.info("History:   http://localhost:8000/history/H1?hours=24")
    logger.info("Health:    http://localhost:8000/health")
    logger.info("=" * 60)
    cmd_api()


COMMANDS = {
    "api":      cmd_api,
    "simulate": cmd_simulate,
    "train":    cmd_train,
    "infer":    cmd_infer,
    "spark":    cmd_spark,
    "demo":     cmd_demo,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
