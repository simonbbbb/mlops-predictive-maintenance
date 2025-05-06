import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import os
import json
from typing import Dict, Tuple
import great_expectations as ge

def generate_sensor_data(equipment_id, days=30, frequency_minutes=5, failure_probability=0.05):
    """Generate simulated sensor data with occasional anomalies."""
    # Create timestamp range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{frequency_minutes}min')
    data = {
        'timestamp': timestamps,
        'equipment_id': equipment_id,
        'temperature': [],
        'vibration': [],
        'pressure': [],
        'noise_level': [],
        'failure': []
    }
    normal_temp = 75.0
    normal_vibration = 0.5
    normal_pressure = 100.0
    normal_noise = 60.0
    for i in range(len(timestamps)):
        will_fail = random.random() < failure_probability
        failure_countdown = random.randint(10, 30) if will_fail else -1
        failure_detected = False
        for j in range(i, min(i + failure_countdown, len(timestamps))):
            if failure_countdown > 0:
                progression = (j - i) / failure_countdown
                anomaly_factor = progression * 2.0
            else:
                anomaly_factor = 0.0
            temp = normal_temp + np.random.normal(0, 1) + (anomaly_factor * 15)
            vibration = normal_vibration + np.random.normal(0, 0.1) + (anomaly_factor * 0.8)
            pressure = normal_pressure + np.random.normal(0, 2) + (anomaly_factor * -10)
            noise = normal_noise + np.random.normal(0, 3) + (anomaly_factor * 20)
            if j == i + failure_countdown - 1:
                failure_detected = True
            if j == i:
                data['temperature'].append(temp)
                data['vibration'].append(vibration)
                data['pressure'].append(pressure)
                data['noise_level'].append(noise)
                data['failure'].append(1 if failure_detected else 0)
    df = pd.DataFrame(data)
    os.makedirs('data/raw', exist_ok=True)
    filename = f'data/raw/sensor_data_equipment_{equipment_id}_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(filename, index=False)
    return filename

class DataPreprocessor:
    def __init__(self, config_path: str = "config/data_processing.json"):
        """Initialize the data preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/validation_results', exist_ok=True)

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate the input data using Great Expectations."""
        ge_df = ge.from_pandas(df)
        validation_results = {}
        expected_columns = self.config['expected_columns']
        validation_results['has_required_columns'] = all(col in df.columns for col in expected_columns)
        validation_results['missing_values'] = df[expected_columns].isnull().sum().to_dict()
        for col, ranges in self.config['value_ranges'].items():
            if col in df.columns:
                validation_results[f'{col}_in_range'] = ge_df.expect_column_values_to_be_between(
                    col, ranges['min'], ranges['max']
                ).success
        validation_results['no_duplicate_timestamps'] = not df.duplicated(
            subset=['equipment_id', 'timestamp']
        ).any()
        validation_success = all([
            validation_results['has_required_columns'],
            validation_results['no_duplicate_timestamps'],
            all(validation_results[f'{col}_in_range'] for col in self.config['value_ranges'] if col in df.columns)
        ])
        return validation_success, validation_results

    def preprocess_data(self, input_path: str) -> str:
        df = pd.read_csv(input_path)
        is_valid, validation_results = self.validate_data(df)
        validation_file = f'data/validation_results/validation_{os.path.basename(input_path)}.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        if not is_valid:
            print(f"Data validation failed for {input_path}. See {validation_file} for details.")
            return None
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if self.config.get('feature_engineering', {}).get('rolling_windows', False):
            window_sizes = self.config['feature_engineering']['window_sizes']
            for col in ['temperature', 'vibration', 'pressure', 'noise_level']:
                for window in window_sizes:
                    df[f'{col}_rolling_mean_{window}'] = df.groupby('equipment_id')[col].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(level=0, drop=True)
                    df[f'{col}_rolling_std_{window}'] = df.groupby('equipment_id')[col].rolling(
                        window=window, min_periods=1
                    ).std().reset_index(level=0, drop=True).fillna(0)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        if 'prediction_window' in self.config:
            window = self.config['prediction_window']
            df['failure_within_window'] = df.groupby('equipment_id')['failure'].rolling(
                window=window, min_periods=1
            ).max().reset_index(level=0, drop=True)
        output_path = f'data/processed/processed_{os.path.basename(input_path)}'
        df.to_csv(output_path, index=False)
        return output_path

if __name__ == "__main__":
    # Generate data
    for equipment_id in range(1, 6):
        filename = generate_sensor_data(equipment_id)
        print(f"Generated data saved to {filename}")
    # Process all files in the raw directory
    if not os.path.exists('config/data_processing.json'):
        print("Missing config/data_processing.json. Please create it before running preprocessing.")
    else:
        preprocessor = DataPreprocessor()
        raw_files = [os.path.join('data/raw', f) for f in os.listdir('data/raw') if f.endswith('.csv')]
        for file in raw_files:
            output_path = preprocessor.preprocess_data(file)
            if output_path:
                print(f"Processed {file} -> {output_path}")
            else:
                print(f"Failed to process {file}") 