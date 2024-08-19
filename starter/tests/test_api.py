
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@pytest.fixture()
def offline_request_1():
    """label <=50K from data"""
    request = {
        "age":39,
        "workclass":"State-gov",
        "fnlgt":77516,
        "education":"Bachelors",
        "education_num":13,
        "marital_status":"Never-married",
        "occupation":"Adm-clerical",
        "relationship":"Not-in-family",
        "race":"White",
        "sex":"Male",
        "capital_gain":2174,
        "capital_loss":0,
        "hours_per_week":40,
        "native_country":"United-States"
    }
    return request


@pytest.fixture()
def offline_request_2():
    """label >=50K from data"""
    request = {
        "age":52,
        "workclass":"Self-emp-not-inc",
        "fnlgt":209642,
        "education":"HS-grad",
        "education_num":9,
        "marital_status":"Married-civ-spouse",
        "occupation":"Exec-managerial",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital_gain":0,
        "capital_loss":0,
        "hours_per_week":45,
        "native_country":"United-States"
    }
    return request


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the project!"}

def test_post_1(offline_request_1):
    response = client.post("/inference", json=offline_request_1)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">=50K"]


def test_post_2(offline_request_2):
    response = client.post("/inference", json=offline_request_2)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">=50K"]

