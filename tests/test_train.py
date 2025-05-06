import os

def test_model_artifacts_exist():
    model_dir = 'models'
    metadata_dir = 'models/metadata'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    metadata_files = [f for f in os.listdir(metadata_dir) if f.startswith('metadata_') and f.endswith('.json')]
    assert len(model_files) > 0, "No model .pkl files found."
    assert len(metadata_files) > 0, "No model metadata files found." 