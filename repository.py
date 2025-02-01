import logging
from dagster import repository, job
from solids.data_preprocessing import read_and_clean_data
from solids.feature_engineering import create_spreads_and_more
from solids.model_training import train_kNN, train_GBM, train_RF

# Set up logging
import os
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
    # Log and run the data preprocessing step
    try:
        logging.info("Running data preprocessing.")    
        raw_data = read_and_clean_data()
    
        # Log and pass the raw_data output from data preprocessing to feature engineering
        logging.info("Running feature engineering.")
        spread_data, features = create_spreads_and_more(raw_data)

        # Log and pass the engineered features to model training
        logging.info("Running model training for kNN.")
        train_kNN(features)
        train_GBM(features)
        train_RF(features)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

    logging.info("Pipeline execution finished.")

# Define the repository
@repository
def my_repository():
    return [stock_data_pipeline]  