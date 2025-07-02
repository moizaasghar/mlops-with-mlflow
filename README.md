# MLOps Sentiment Analysis Project

## Project Overview

This project demonstrates a complete MLOps pipeline for sentiment analysis using MLflow for experiment tracking and model management. The system includes model training, evaluation, deployment, and serving capabilities with both REST API and web interface.

## Features

- **Dual Model Training**: Compare "good" vs "bad" baseline models
- **MLflow Integration**: Complete experiment tracking and model registry
- **FastAPI Backend**: RESTful API for model serving
- **Gradio Frontend**: Interactive web interface for testing
- **Automated Model Download**: Download models from MLflow registry for deployment

## Project Structure

```
mlops/
├── requirements.txt          # Python dependencies
├── train_good_model.py      # Full dataset training with optimized hyperparameters
├── train_bad_model.py       # Baseline model with minimal features
├── .gitignore              # Git ignore file
└── app/
    ├── client/
    │   └── main.py         # Gradio web interface
    └── server/
        ├── main.py         # FastAPI REST API server
        ├── model.py        # Model loading and prediction logic
        └── download.py     # Download models from MLflow registry
```

## Models

### Good Model
- **Dataset**: Full multiclass sentiment analysis dataset
- **Features**: TF-IDF with unigrams + bigrams
- **Algorithm**: Logistic Regression with 'saga' solver
- **Purpose**: Production-ready model with optimized performance

### Bad Model (Baseline)
- **Dataset**: Only 10% of training data
- **Features**: TF-IDF with single most frequent token
- **Algorithm**: Logistic Regression with 1 iteration
- **Purpose**: Weak baseline for comparison

## Setup and Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MLflow Server

```bash
mlflow server --host 127.0.0.1 --port 8080
```

The MLflow UI will be available at: http://127.0.0.1:8080

## Usage

### Training Models

#### Train the Good Model
```bash
python train_good_model.py
```

#### Train the Bad Model
```bash
python train_bad_model.py
```

Both scripts will:
- Load the dataset from Hugging Face
- Train the respective models
- Log metrics, parameters, and artifacts to MLflow
- Register the model in MLflow Model Registry

### Model Deployment

#### 1. Download Model from MLflow Registry

```bash
cd app/server
python download.py
```

This downloads the registered model to `./app/server/model/` directory.

#### 2. Start the FastAPI Backend

```bash
cd app/server
python main.py
```

The API will be available at: http://127.0.0.1:8000

#### 3. Start the Gradio Frontend (Optional)

```bash
cd app/client
python main.py
```

The web interface will be available at: http://127.0.0.1:7860

### API Testing

Test the sentiment analysis API using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"I love this product!\"}"
```

Expected response:
```json
{
  "sentiment": "Positive"
}
```

## API Endpoints

### POST /predict

Predicts sentiment for input text.

**Request Body:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "sentiment": "Positive|Negative|Neutral"
}
```

## MLflow Experiment Tracking

The project uses MLflow for comprehensive experiment tracking:

- **Experiments**: All runs are logged under "sentiment-analysis" experiment
- **Metrics**: Accuracy, confusion matrix visualizations
- **Parameters**: Model hyperparameters, dataset configuration
- **Artifacts**: Trained models, plots, data splits
- **Model Registry**: Trained models are registered for easy deployment

## Technologies Used

- **MLflow**: Experiment tracking and model management
- **FastAPI**: REST API framework
- **Gradio**: Web interface for model interaction
- **Scikit-learn**: Machine learning pipeline
- **Transformers**: Dataset loading and preprocessing
- **Uvicorn**: ASGI server for FastAPI
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization

## Development Workflow

1. **Experiment**: Train models with different configurations
2. **Track**: Monitor experiments in MLflow UI
3. **Compare**: Evaluate model performance across runs
4. **Register**: Register best models in MLflow Model Registry
5. **Deploy**: Download and serve models via API
6. **Test**: Validate deployment with test requests

## Model Performance

The project includes visualization of:
- Training/validation accuracy
- Confusion matrices
- Feature importance (TF-IDF weights)
- Model comparison metrics

All visualizations are automatically logged to MLflow for easy comparison.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

This project is open source and available under the MIT License.
