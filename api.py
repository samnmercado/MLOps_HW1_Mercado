from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel

app = FastAPI()

# Define the input structure for the model
class InputData(BaseModel):
    param1: int

# Load the MLflow model (replace with your registered model name)
model = mlflow.pyfunc.load_model("models:/your_model_name/1")

@app.post("/predict")
def predict(data: InputData):
    # Prepare input data
    input_data = [[data.param1]]
    # Make prediction
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
