# Put the code for your API here.

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import logging

from modules.ml import inference, process_data


class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 23,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 8071,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States"
                }
            ]
        }
    }


with open("model/logistic_regression.pkl", "rb") as f:
    model_dict = pickle.load(f)
    MODEL = model_dict["model"]
    ENCODER = model_dict["encoder"]
    LB = model_dict["lb"]


app = FastAPI()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@app.get(path="/")
def welcome_root():
    return {"message": "Welcome to the project!"}


@app.post(path="/inference")
def app_inference(request:InferenceRequest):
    """
    Perform inference on the provided request.

    Args:
        request (InferenceRequest): The input for inference.

    Returns:
        dict: The inference result.
    """
    logger.info(f"REQUEST: {request}")
    request_dict = request.dict()
    logger.info(f"REQUEST_DICT: {request_dict}")
    df = pd.DataFrame([request_dict])
    X, _, _, _ = process_data(
        X=df,
        categorical_features=[
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country"
        ],
        training=False,
        encoder=ENCODER,
        lb=LB
    )
    logger.info(f"INPUT ARRAY: {X}")
    preds = inference(model=MODEL, X=X)
    result = {
        "prediction": LB.inverse_transform(preds)[0]
    }
    return result


if __name__ == "__main__":

    uvicorn.run(app="main:app", host="0.0.0.0", port=5000)
