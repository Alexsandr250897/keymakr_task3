from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import joblib

from train import train_model_from_df
from create_tasks import generate_csv_file

app = FastAPI()

model = None


@app.post("/generate-csv")
def generate_csv():
    path = generate_csv_file()
    return {"message": "CSV created", "file": path}



@app.post("/train")
async def train(file: UploadFile = File(...)):
    global model

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    model = train_model_from_df(df)

    return {"message": "Model trained"}



class TaskRequest(BaseModel):
    description: str


@app.post("/predict")
def predict(task: TaskRequest):
    global model
    if model is None:
        model = joblib.load("model.pkl")

    prediction = model.predict([task.description])[0]

    return {
        "prediction": prediction
    }