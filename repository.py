from dagster import repository, job
from solids.data_preprocessing import data_preprocessing
from solids.feature_engineering import feature_engineering
from solids.model_training import model_training

# Define the pipeline
@job
def stock_data_pipeline():
    # Run the data preprocessing step
    raw_data = data_preprocessing()
    
    # Pass the raw_data output from data preprocessing to feature engineering
    features = feature_engineering(raw_data)
    
    # Pass the engineered features to model training
    model = model_training(features)

# Define the repository
@repository
def my_repository():
    return [stock_data_pipeline]
