"""
Script to train and log a BAD (weak baseline) sentiment classification model using MLflow.
This model is trained on only 10% of the data, with minimal TF-IDF features (unigram only)
and a single iteration of LogisticRegression, to serve as a low-quality baseline.
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
logger.info("Starting BAD model training...")

# -----------------------------------------------------------------------------
# 2. Define Constants and Paths
# -----------------------------------------------------------------------------
DATA_DIR       = "data"         # where CSV splits will be stored
ARTIFACTS_DIR  = "artifacts"    # where plots and model artifacts will be stored
DATASET_NAME   = "Sp1786/multiclass-sentiment-analysis-dataset"
EXPERIMENT     = "sentiment-analysis"
TRAIN_SPLIT    = "sentiment-train-bad"
EVAL_SPLIT     = "sentiment-eval"

# Fraction of full training data to use for the BAD model
BAD_TRAIN_FRAC = 0.1

# TF-IDF and LogisticRegression hyperparameters for the BAD baseline
TFIDF_MAX_FEAT  = 1             # only the single most frequent token
TFIDF_NGRAM     = (1, 1)        # unigrams only
LOGREG_MAX_ITER = 1             # only one iteration
LOGREG_SOLVER   = "saga"

# Train/test split parameters
TEST_SIZE    = 0.2              # 20% held out for testing
RANDOM_STATE = 42               # fixed seed for reproducibility

# Ensure output directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Load and Prepare Data
# -----------------------------------------------------------------------------
# Download the 'train' split from Hugging Face
dataset = load_dataset(DATASET_NAME, split="train")
texts   = dataset["text"]      # list of input sentences
labels  = dataset["label"]     # list of integer labels (0/1/2)

# Perform a fixed train/test split to maintain comparability
X_train_full, X_test, y_train_full, y_test = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels            # keep label distribution consistent
)

# Convert to pandas.DataFrame for MLflow logging
df_train_full = pd.DataFrame({"text": X_train_full, "label": y_train_full})
df_test       = pd.DataFrame({"text": X_test,       "label": y_test})

# Sample only a fraction of the training data for the BAD model
df_train_small = df_train_full.sample(frac=BAD_TRAIN_FRAC, random_state=RANDOM_STATE)

# Persist CSV files for record-keeping and MLflow data ingestion
train_csv = os.path.join(DATA_DIR, "train_bad.csv")
test_csv  = os.path.join(DATA_DIR, "test.csv")
df_train_small.to_csv(train_csv, index=False)
df_test.to_csv(test_csv,      index=False)

# -----------------------------------------------------------------------------
# 4. Log Datasets to MLflow
# -----------------------------------------------------------------------------
# Track the exact dataset versions used for training and evaluation
train_dataset = mlflow.data.from_pandas(
    df_train_small,
    source=train_csv,
    name=TRAIN_SPLIT,
    targets="label"
)
eval_dataset = mlflow.data.from_pandas(
    df_test,
    source=test_csv,
    name=EVAL_SPLIT,
    targets="label"
)

# -----------------------------------------------------------------------------
# 5. Define the ML Pipeline
# -----------------------------------------------------------------------------
pipeline = Pipeline([
    # Convert text to a single-feature TF-IDF vector
    ("tfidf", TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        ngram_range=TFIDF_NGRAM
    )),
    # Train LogisticRegression for one iteration as a weak learner
    ("clf", LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        solver=LOGREG_SOLVER
    ))
])

# -----------------------------------------------------------------------------
# 6. Train, Evaluate, Visualize, and Log to MLflow
# -----------------------------------------------------------------------------
with tqdm(total=3, desc="Training BAD Model") as pbar:
    # 6.1 Fit the pipeline on the small training subset
    pipeline.fit(df_train_small["text"], df_train_small["label"])
    pbar.update(1)

    # 6.2 Predict on the test set and calculate accuracy
    y_pred = pipeline.predict(df_test["text"])
    acc    = accuracy_score(df_test["label"], y_pred)
    pbar.update(1)

    # 6.3 Generate confusion matrix and save as plot
    cm   = confusion_matrix(df_test["label"], y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Neutral", "Positive"]
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - BAD Model")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_bad.png")
    plt.savefig(cm_path)
    plt.close()

    # 6.4 Prepare example input and model signature for logging
    input_example = pd.DataFrame({"text": [df_test["text"].iloc[0]]})
    signature     = infer_signature(df_test[["text"]], y_pred[:1])

    # 6.5 Configure MLflow and log everything in a single run
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="Train Bad Model"):
        # Log datasets (lineage)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(eval_dataset,  context="evaluation")

        # Log parameters describing the BAD baseline
        mlflow.log_param("data_used",  f"{int(BAD_TRAIN_FRAC * 100)}% of training split")
        mlflow.log_param("model_type", "Tfidf+LogReg+SAGA (BAD)")

        # Log evaluation metric
        mlflow.log_metric("accuracy", acc)

        # Log the trained sklearn pipeline as an MLflow model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="SentimentClassifier",
            input_example=input_example,
            signature=signature
        )

        # Log the confusion matrix plot under the 'plots' artifact folder
        mlflow.log_artifact(cm_path, artifact_path="plots")

    pbar.update(1)

# Final log message indicating completion and performance
logger.info(f"BAD model training complete! Accuracy: {acc:.4f}")
