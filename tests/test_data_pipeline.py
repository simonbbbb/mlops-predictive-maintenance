import os
import pandas as pd

def test_processed_data_exists():
    processed_dir = 'data/processed'
    files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    assert len(files) > 0, "No processed data files found."

    # Check columns in the first processed file
    df = pd.read_csv(os.path.join(processed_dir, files[0]))
    expected_columns = [
        "timestamp", "equipment_id", "temperature", "vibration", "pressure", "noise_level", "failure",
        "temperature_rolling_mean_3", "temperature_rolling_std_3",
        "vibration_rolling_mean_3", "vibration_rolling_std_3",
        "pressure_rolling_mean_3", "pressure_rolling_std_3",
        "noise_level_rolling_mean_3", "noise_level_rolling_std_3",
        "hour", "day_of_week", "failure_within_window"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}" 