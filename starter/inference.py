import requests
import pickle


# Request to the Render server
url = "https://udacity-mlops-project3-9zpf.onrender.com"
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
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

response = requests.post(
    url=url,
    json=data
)
print(f"Status code: {response.status_code}")
print(response.json())