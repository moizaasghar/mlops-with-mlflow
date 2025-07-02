"""
 GOOD Sentiment Model Training with MLflow

This script trains a “GOOD” sentiment classification model on the full dataset 
using TF-IDF features (unigrams + bigrams) and a Logistic Regression classifier
with the 'saga' solver. It then evaluates, visualizes, and logs all artifacts 
and metrics to MLflow for reproducibility and deployment.
"""

import logging
import os
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Configure Logging
# -----------------------------------------------------------------------------
# - level=logging.INFO: show informational messages
# - format includes timestamp, log level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Starting GOOD model training...")

# -----------------------------------------------------------------------------
# 2. Define Constants and Paths
# -----------------------------------------------------------------------------
# Directory names for storing data and artifacts
DATA_DIR      = "data"        # CSV files saved here
ARTIFACTS_DIR = "artifacts"   # plots and other files saved here

# Dataset and experiment identifiers
DATASET_NAME     = "Sp1786/multiclass-sentiment-analysis-dataset"
EXPERIMENT_NAME  = "sentiment-analysis"
TRAIN_SPLIT_NAME = "sentiment-train-good"
EVAL_SPLIT_NAME  = "sentiment-eval"

# Model hyperparameters
TFIDF_MAX_FEAT  = 3000             # limit vocabulary to top 3000 features
TFIDF_NGRAM     = (1, 2)           # include unigrams and bigrams
LOGREG_MAX_ITER = 300              # maximum iterations for solver convergence
LOGREG_SOLVER   = "saga"           # solver that supports L1 and multinomial

# Train/test split settings
TEST_SIZE    = 0.2                 # 20% of data held out for evaluation
RANDOM_STATE = 42                  # seed for reproducibility

# Ensure output directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Load and Split Data
# -----------------------------------------------------------------------------
# Download the 'train' split from Hugging Face
dataset = load_dataset(DATASET_NAME, split="train")
texts   = dataset["text"]          # raw input sentences
labels  = dataset["label"]         # integer labels (0/1/2)

# Create a fixed train/test split to ensure comparability
X_train_full, X_test, y_train_full, y_test = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels                # preserve class distribution
)

# Convert to pandas for MLflow compatibility
df_train_full = pd.DataFrame({"text": X_train_full, "label": y_train_full})
df_test       = pd.DataFrame({"text": X_test,       "label": y_test})

# Save splits locally (for record-keeping and MLflow data ingestion)
train_csv = os.path.join(DATA_DIR, "train_good.csv")
eval_csv  = os.path.join(DATA_DIR, "test.csv")
df_train_full.to_csv(train_csv, index=False)
df_test.to_csv(eval_csv,      index=False)

# -----------------------------------------------------------------------------
# 4. Log Datasets to MLflow
# -----------------------------------------------------------------------------
# Creating MLflow Data artifacts tracks dataset versions and lineage
train_dataset = mlflow.data.from_pandas(
    df_train_full,
    source=train_csv,
    name=TRAIN_SPLIT_NAME,
    targets="label"
)
eval_dataset = mlflow.data.from_pandas(
    df_test,
    source=eval_csv,
    name=EVAL_SPLIT_NAME,
    targets="label"
)

# -----------------------------------------------------------------------------
# 5. Define the Modeling Pipeline
# -----------------------------------------------------------------------------
# - TF-IDF vectorizer: transforms text into sparse feature matrix
# - Logistic Regression: binary/multiclass classifier on TF-IDF features
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        ngram_range=TFIDF_NGRAM
    )),
    ("clf", LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        solver=LOGREG_SOLVER
    ))
])

# -----------------------------------------------------------------------------
# 6. Train, Evaluate, Visualize, and Log Everything
# -----------------------------------------------------------------------------
with tqdm(total=3, desc="Training GOOD Model") as pbar:
    # 6.1 Fit the pipeline on the training data
    pipeline.fit(df_train_full["text"], df_train_full["label"])
    pbar.update(1)

    # 6.2 Predict on the test set and calculate accuracy
    y_pred = pipeline.predict(df_test["text"])
    acc    = accuracy_score(df_test["label"], y_pred)
    pbar.update(1)

    # 6.3 Create and save a confusion matrix plot
    cm   = confusion_matrix(df_test["label"], y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Neutral", "Positive"]
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix - GOOD Model")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_good.png")
    plt.savefig(cm_path)
    plt.close()

    # 6.4 Prepare an example input and infer the model signature
    input_example = pd.DataFrame({"text": [df_test["text"].iloc[0]]})
    signature     = infer_signature(df_test[["text"]], y_pred[:1])

    # 6.5 Configure MLflow and start a new run for logging
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="Train Good Model"):
        # Log data artifacts
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(eval_dataset,  context="evaluation")

        # Log hyperparameters
        mlflow.log_param("data_used",  "100% of training split")
        mlflow.log_param("model_type", "Tfidf+LogReg+SAGA (GOOD)")

        # Log performance metric
        mlflow.log_metric("accuracy", acc)

        # Log the trained pipeline as an MLflow model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="SentimentClassifier",
            input_example=input_example,
            signature=signature
        )

        # Log the confusion matrix plot under the 'plots' artifact directory
        mlflow.log_artifact(cm_path, artifact_path="plots")

    pbar.update(1)

# Final log to indicate completion and recorded accuracy
logger.info(f"GOOD model training complete! Accuracy: {acc:.4f}")
