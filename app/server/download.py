"""
This script downloads a registered MLflow model artifact from the MLflow Model Registry
to a local directory, so it can be loaded and used without requiring a live MLflow server
at inference time.

Steps:
1. Configure MLflow tracking URI to point at the MLflow server.
2. Use the MLflow Artifacts API to fetch the model files for a given registry entry.
3. Print the local path where the model artifacts were saved.
"""

import mlflow
import mlflow.artifacts

# Point MLflow client at the local tracking server.
mlflow.set_tracking_uri("http://127.0.0.1:8080/")

# Use the Artifacts API to download all files associated with the specified model.
# - artifact_uri: the model registry URI in the form "models:/<ModelName>@<Aliases>".
# - dst_path:    the local filesystem directory where artifacts will be saved.
local_model_dir = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/SentimentClassifier@production",
    dst_path="./app/server/model"
)

# Print the path to which the model artifacts were downloaded.
print("Model downloaded to:", local_model_dir)
