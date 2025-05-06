import requests
import pytest

BASE_URL = "http://localhost:8000"

@pytest.mark.order(1)
def test_root():
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"

@pytest.mark.order(2)
def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code in [200, 503]  # 503 if model not loaded
    assert "status" in resp.json()

@pytest.mark.order(3)
def test_predict():
    # Example payload (should match SensorData schema)
    payload = {
        "equipment_id": 1,
        "timestamp": "2023-05-01T10:00:00",
        "temperature": 76.2,
        "vibration": 0.54,
        "pressure": 98.6,
        "noise_level": 62.3
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    # Accept 200 (success) or 503 (model not loaded)
    assert resp.status_code in [200, 503]
    if resp.status_code == 200:
        data = resp.json()
        assert "prediction_id" in data
        assert "failure_probability" in data
        assert "prediction" in data 