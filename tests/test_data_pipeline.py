import os
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from pipeline.data_pipeline import generate_sensor_data

@pytest.fixture(scope="module", autouse=True)
def ensure_data_directories():
    """Ensure data directories exist"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/validation_results', exist_ok=True)

@pytest.mark.order(1)
def test_generate_data():
    """Test data generation function"""
    # Generate a single equipment data
    filename = generate_sensor_data(equipment_id=999, days=1, frequency_minutes=15)
    assert os.path.exists(filename)
    df = pd.read_csv(filename)
    assert len(df) > 0
    assert "equipment_id" in df.columns
    assert "timestamp" in df.columns
    assert "temperature" in df.columns
    assert "vibration" in df.columns
    assert "pressure" in df.columns
    assert "noise_level" in df.columns
    assert "failure" in df.columns

@pytest.mark.order(2)
def test_processed_data_exists():
    processed_dir = 'data/processed'
    
    # Skip if no processed files exist (will be created by pipeline run in CI)
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        pytest.skip("No processed data files found. Run data pipeline first.")
        
    files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    assert len(files) > 0, "No processed data files found."

    # Check columns in the first processed file
    df = pd.read_csv(os.path.join(processed_dir, files[0]))
    expected_columns = [
        "timestamp", "equipment_id", "temperature", "vibration", "pressure", "noise_level", "failure"
    ]
    
    # Check at least the basic columns exist
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}" 