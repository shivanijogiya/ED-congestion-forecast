"""
Integration tests for the FastAPI server.
Uses TestClient — no real infrastructure needed.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestForecastEndpoint:
    def test_forecast_returns_200_for_valid_hospital(self):
        resp = client.get("/forecast/H1")
        assert resp.status_code == 200

    def test_forecast_has_required_fields(self):
        resp = client.get("/forecast/H1")
        data = resp.json()
        assert "hospital_id" in data
        assert "hospital_severity" in data
        assert "departments" in data
        assert "prediction_timestamp" in data
        assert "model_info" in data

    def test_forecast_department_has_horizons(self):
        resp = client.get("/forecast/H1")
        dept = resp.json()["departments"][0]
        assert "forecasts" in dept
        assert "1h" in dept["forecasts"]
        assert "2h" in dept["forecasts"]
        assert "4h" in dept["forecasts"]
        assert "8h" in dept["forecasts"]

    def test_forecast_404_for_unknown_hospital(self):
        resp = client.get("/forecast/UNKNOWN")
        assert resp.status_code == 404

    def test_forecast_severity_valid_values(self):
        resp = client.get("/forecast/H2")
        data = resp.json()
        assert data["hospital_severity"] in ("green", "amber", "red")
        for dept in data["departments"]:
            assert dept["severity_label"] in ("green", "amber", "red")

    def test_forecast_congestion_scores_in_01(self):
        resp = client.get("/forecast/H3")
        for dept in resp.json()["departments"]:
            for label, score in dept["forecasts"].items():
                assert 0 <= score <= 1, f"Score {score} out of [0,1] for {label}"


class TestHospitalsEndpoint:
    def test_returns_all_hospitals(self):
        resp = client.get("/hospitals")
        data = resp.json()
        assert data["total"] == 6
        assert len(data["hospitals"]) == 6

    def test_hospitals_have_departments_and_edges(self):
        resp = client.get("/hospitals")
        for h in resp.json()["hospitals"]:
            assert len(h["departments"]) > 0
            assert len(h["edges"]) > 0

    def test_hospital_edge_types_valid(self):
        resp = client.get("/hospitals")
        for h in resp.json()["hospitals"]:
            for edge in h["edges"]:
                assert edge["edge_type"] in ("TRANSFER", "SHARED_RESOURCE", "PROXIMITY")


class TestHistoryEndpoint:
    def test_returns_timeline(self):
        resp = client.get("/history/H1?hours=24")
        data = resp.json()
        assert len(data["timeline"]) == 24

    def test_timeline_values_in_range(self):
        resp = client.get("/history/H1?hours=12")
        for entry in resp.json()["timeline"]:
            for dept_id, occ in entry["departments"].items():
                assert 0 <= occ <= 1


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_all_components(self):
        data = resp = client.get("/health").json()
        components = data["components"]
        for key in ("kafka", "cassandra", "elasticsearch", "spark", "model"):
            assert key in components
