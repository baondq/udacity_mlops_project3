import requests
import pickle


# Request to the Render server
url = "https://udacity-mlops-project3-9zpf.onrender.com/inference"
with open("model/logistic_regression.pkl", "rb") as f:
    model_dict = pickle.load(f)
    MODEL = model_dict["model"]
    ENCODER = model_dict["encoder"]
    LB = model_dict["lb"]
data = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States"
}
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

response = requests.post(
    url=url,
    json=data
)
print(f"Status code: {response.status_code}")
print(response.json())