# MLOps Predictive Maintenance System

![MLOps Pipeline](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-teal)
![ML](https://img.shields.io/badge/ML-Predictive%20Maintenance-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)

A comprehensive end-to-end MLOps pipeline for predictive maintenance, demonstrating modern machine learning operations practices. This portfolio project simulates a real-world industrial use case where sensor data is used to predict equipment failures before they occur.

## Project Overview

### Purpose and Problem Statement
This project demonstrates a complete MLOps lifecycle for predictive maintenance in manufacturing/industrial scenarios. It addresses the common challenge of detecting equipment failures before they happen, allowing for proactive maintenance scheduling and reducing costly downtime.

### Key Features
- **End-to-End Pipeline**: From data generation to model serving with monitoring
- **Real-Time Predictions**: FastAPI endpoint for real-time failure predictions
- **Robust Data Validation**: Ensures data quality with Pandera schema validation
- **Experiment Tracking**: MLflow integration for model versioning and experiment tracking
- **Automated CI/CD**: GitHub Actions for testing and deployment
- **Containerized Deployment**: Docker for consistent deployment
- **Monitoring**: Data drift and model performance monitoring

## Technical Architecture

```
├── data/                      # Data storage and versioning
│   ├── raw/                   # Raw sensor data
│   ├── processed/             # Processed datasets
│   └── validation_results/    # Data validation reports
├── model/                     # Model definitions and training
│   └── train.py               # Model training script with MLflow tracking
├── pipeline/                  # Pipeline definitions
│   └── data_pipeline.py       # Data generation and processing pipeline
├── api/                       # API service
│   └── main.py                # FastAPI application for model serving
├── monitoring/                # Monitoring components
├── infrastructure/            # Infrastructure as code
├── tests/                     # Automated tests
│   ├── test_api.py            # API endpoint tests
│   ├── test_data_pipeline.py  # Data processing tests
│   └── test_train.py          # Model training tests
├── .github/workflows/         # CI/CD with GitHub Actions
└── requirements.txt           # Project dependencies
```

## Technologies Used

### Core Components
- **Python**: Primary programming language
- **Pandas/NumPy**: Data manipulation and processing
- **Scikit-learn**: Machine learning models (RandomForest and GradientBoosting)
- **FastAPI**: Modern API framework for model serving
- **Pandera**: Data validation and schema enforcement

### MLOps Tools
- **MLflow**: Experiment tracking, model versioning, and artifact storage
- **GitHub Actions**: CI/CD automation
- **Docker**: Containerization for consistent deployment
- **pytest**: Automated testing framework

## Implementation Details

### 1. Data Pipeline
The data pipeline simulates equipment sensor readings (temperature, vibration, pressure, noise) with occasional anomalies that predict failures:

```python
# Data generation with occasional failure patterns
def generate_sensor_data(equipment_id, days=30, frequency_minutes=5, failure_probability=0.05):
    # Simulate normal conditions with occasional degradation patterns
    # As components approach failure, sensor readings show increasing anomalies
    # Returns timestamped sensor readings with failure flags
```

### 2. Data Validation
All data undergoes strict validation to ensure quality:

```python
# Schema-based validation with Pandera
schema = DataFrameSchema({
    "timestamp": Column(pa.DateTime),
    "equipment_id": Column(pa.Int),
    "temperature": Column(pa.Float, Check.in_range(50, 120)),
    "vibration": Column(pa.Float, Check.in_range(0, 5)),
    "pressure": Column(pa.Float, Check.in_range(50, 200)),
    "noise_level": Column(pa.Float, Check.in_range(30, 120)),
})
```

### 3. Feature Engineering
Time-series features are engineered to capture equipment degradation patterns:

- Rolling statistics (mean/std) over different time windows
- Time-based features (hour of day, day of week)
- Future failure probability (used as target variable)

### 4. Model Training with MLflow
Models are trained with experiment tracking:

```python
with mlflow.start_run() as run:
    # Log parameters, metrics, artifacts
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    mlflow.sklearn.log_model(model, "model")
```

### 5. Model Serving API
A FastAPI application provides real-time prediction endpoints:

```
GET  /health                # Health check and model info
GET  /model/info            # Model metadata and performance metrics
POST /predict               # Single prediction endpoint
POST /predict/batch         # Batch prediction endpoint
```

### 6. CI/CD Pipeline
Automated testing and deployment with GitHub Actions:

- Runs data pipeline in the CI environment
- Trains a new model for testing
- Starts the API server and runs end-to-end tests
- Reports code coverage

## Getting Started

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/simonbbbb/mlops-predictive-maintenance.git
   cd mlops-predictive-maintenance
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Generate and process data:**
   ```bash
   python pipeline/data_pipeline.py
   ```

4. **Train the model:**
   ```bash
   python model/train.py
   ```

5. **Start the API server:**
   ```bash
   uvicorn api.main:app --reload
   ```

6. **Access the API documentation:**
   Open your browser to http://localhost:8000/docs

### Running Tests
```bash
pytest tests/
```

## Learning Outcomes

This project demonstrates several key MLOps practices:

1. **Reproducible ML Pipelines**: Structured data processing and model training
2. **Experiment Tracking**: Version control for ML models and experiments
3. **Model Deployment**: Serving ML models through APIs
4. **Monitoring**: Detecting data drift and model performance degradation
5. **Testing**: Comprehensive test suite for all components
6. **CI/CD**: Automated testing and deployment
7. **Infrastructure as Code**: Containerization and deployment configuration

## Future Enhancements

- Model A/B testing framework
- Online learning capabilities
- Feature store integration
- Extended monitoring dashboard
- Kubernetes deployment
- Automated retraining pipeline

## Portfolio Demonstration

This project showcases applied knowledge in:
- Machine Learning Engineering
- Software Engineering best practices
- DevOps and CI/CD
- API Development
- Data Engineering

---

*This project is part of my portfolio to demonstrate MLOps skills and best practices. It's designed as a learning tool and template for real-world predictive maintenance applications.* 