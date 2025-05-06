import pandas as pd
import numpy as np
import os
import json
import pickle
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = "config/model_training.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/metadata', exist_ok=True)
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        processed_dir = 'data/processed'
        processed_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not processed_files:
            logger.error("No processed data files found")
            return None, None, None, None
        dfs = [pd.read_csv(file) for file in processed_files]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        target_col = self.config['target_column']
        feature_cols = self.config['feature_columns']
        if 'exclude_columns' in self.config:
            all_cols = set(combined_df.columns)
            exclude_cols = set(self.config['exclude_columns'])
            feature_cols = list(all_cols - exclude_cols - {target_col})
        X = combined_df[feature_cols]
        y = combined_df[target_col]
        categorical_cols = [col for col in feature_cols if combined_df[col].dtype == 'object']
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=self.config['random_state']
        )
        return X_train, X_test, y_train, y_test

    def train_model(self) -> str:
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        if X_train is None:
            return None
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            mlflow.log_params(self.config['model_params'])
            model_type = self.config['model_type']
            if model_type == 'random_forest':
                model = RandomForestClassifier(**self.config['model_params'])
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(**self.config['model_params'])
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            logger.info(f"Training {model_type} model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"Model performance: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                "true_negatives": int(cm[0][0]),
                "false_positives": int(cm[0][1]),
                "false_negatives": int(cm[1][0]),
                "true_positives": int(cm[1][1])
            }
            mlflow.log_dict(cm_dict, "confusion_matrix.json")
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                mlflow.log_dict(feature_importance, "feature_importance.json")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/predictive_maintenance_{model_type}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.sklearn.log_model(model, "model")
            feature_names = {
                "feature_names": list(X_train.columns)
            }
            feature_names_path = f"models/metadata/feature_names_{timestamp}.json"
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f)
            model_metadata = {
                "model_type": model_type,
                "timestamp": timestamp,
                "mlflow_run_id": run_id,
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "feature_names_path": feature_names_path,
                "model_path": model_path,
                "test_size": self.config['test_size'],
                "random_state": self.config['random_state'],
                "parameters": self.config['model_params']
            }
            metadata_path = f"models/metadata/metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Model metadata saved to {metadata_path}")
            return model_path

if __name__ == "__main__":
    trainer = ModelTrainer()
    model_path = trainer.train_model()
    if model_path:
        print(f"Model training completed successfully. Model saved to {model_path}")
    else:
        print("Model training failed.") 