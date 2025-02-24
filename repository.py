import logging
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from dagster import repository, job, op, graph
from solids.data_preprocessing import read_and_clean_data
from solids.feature_engineering import create_spreads_and_more
from solids.model_training import train_kNN, train_GBM, train_RF

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Wrapper class for MLflow models
class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(model_input)

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/dagster_logs.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log when the repository is being initialized
logging.debug("Repository initialized.")

# Define the pipeline 
@graph
def stock_data_pipeline():
    raw_data = read_and_clean_data()
    X, y = create_spreads_and_more(raw_data)
    train_kNN(X, y)
    train_GBM(X, y)
    train_RF(X, y)

# Define the job
stock_data_pipeline_job = stock_data_pipeline.to_job(name="stock_data_pipeline_job")

# Define the repository
@repository
def my_repository():
    return [stock_data_pipeline_job]