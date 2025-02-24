import logging
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from dagster import repository, job
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
@job
def stock_data_pipeline():
    logging.info("Starting the stock data pipeline execution.")
    try:
        # Log and run the data preprocessing step
        logging.info("Running data preprocessing.")    
        raw_data = read_and_clean_data()
    
        # Log and pass the raw_data output from data preprocessing to feature engineering
        logging.info("Running feature engineering.")
        spread_data, features = create_spreads_and_more(raw_data)

        # Log and pass the engineered features to model training
        logging.info("Running model training for kNN.")
        trained_kNN = train_kNN(features)

        logging.info("Running model training for GBM.")
        trained_GBM = train_GBM(features)

        logging.info("Running model training for RF.")
        trained_RF = train_RF(features)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

    logging.info("Pipeline execution finished.")

# Define the repository
@repository
def my_repository():
    return [stock_data_pipeline]