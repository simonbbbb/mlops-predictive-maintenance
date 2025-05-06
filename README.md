# MLOps Predictive Maintenance System

This project is an end-to-end MLOps pipeline for predictive maintenance, designed for portfolio demonstration. It simulates sensor data, processes and validates it, trains machine learning models, serves predictions via an API, and includes CI/CD and monitoring components.

## Project Architecture

```
├── data/                      # Data storage and versioning
│   ├── raw/                   # Raw sensor data
│   └── processed/             # Processed datasets
├── model/                     # Model definitions and training
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── predict.py             # Prediction functions
├── pipeline/                  # Pipeline definitions
│   ├── data_pipeline.py       # Data processing pipeline
│   ├── training_pipeline.py   # Model training pipeline
│   └── inference_pipeline.py  # Model inference pipeline
├── api/                       # API service
│   ├── main.py                # FastAPI application
│   ├── routers/               # API endpoints
│   └── middleware/            # API middleware (auth, logging)
├── monitoring/                # Monitoring components
│   ├── data_drift.py          # Data drift detection
│   ├── model_drift.py         # Model drift detection
│   └── performance.py         # Performance monitoring
├── infrastructure/            # Infrastructure as code
│   ├── docker/                # Docker configurations
│   ├── k8s/                   # Kubernetes manifests
│   └── terraform/             # Terraform configurations
├── tests/                     # Tests for all components
├── notebooks/                 # Exploratory notebooks
├── .github/                   # CI/CD workflows
│   └── workflows/             # GitHub Actions definitions
├── .gitignore                 # Git ignore file
├── .pre-commit-config.yaml    # Pre-commit hooks
├── pyproject.toml             # Project dependencies
└── README.md                  # Project documentation
```

## Features
- Simulated sensor data generation
- Data validation and preprocessing
- Data versioning with DVC
- Model training and evaluation
- Experiment tracking with MLflow
- FastAPI model serving
- Monitoring and drift detection
- CI/CD with GitHub Actions
- Containerization with Docker

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd MLOps
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Set up DVC remote:**
   ```bash
   dvc remote add -d s3remote s3://your-bucket/dvc-store
   dvc remote modify s3remote endpointurl https://your-endpoint-url
   ```
5. **Run the data generator:**
   ```bash
   python pipeline/data_pipeline.py
   ```
6. **Train the model:**
   ```bash
   python model/train.py
   ```
7. **Start the API:**
   ```bash
   uvicorn api.main:app --reload
   ```

---

For detailed instructions, see each module's README or docstrings. 