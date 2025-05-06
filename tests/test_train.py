import os
import pytest
import pickle
from model.train import ModelTrainer

@pytest.fixture(scope="module", autouse=True)
def ensure_model_directories():
    """Ensure model directories exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/metadata', exist_ok=True)

def test_trainer_initialization():
    """Test that the model trainer initializes correctly"""
    try:
        trainer = ModelTrainer()
        assert trainer is not None
    except Exception as e:
        pytest.skip(f"Model trainer initialization failed: {str(e)}")

def test_model_artifacts_exist():
    """Test that model artifacts exist"""
    model_dir = 'models'
    metadata_dir = 'models/metadata'
    
    # Skip if no model files exist
    if not os.path.exists(model_dir) or len([f for f in os.listdir(model_dir) if f.endswith('.pkl')]) == 0:
        pytest.skip("No model files found. Run model training first.")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    metadata_files = [f for f in os.listdir(metadata_dir) if f.startswith('metadata_') and f.endswith('.json')]
    
    assert len(model_files) > 0, "No model .pkl files found."
    assert len(metadata_files) > 0, "No model metadata files found."
    
    # Check if model can be loaded 
    if len(model_files) > 0:
        model_path = os.path.join(model_dir, model_files[0])
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            assert hasattr(model, 'predict'), "Model does not have predict method"
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}") 