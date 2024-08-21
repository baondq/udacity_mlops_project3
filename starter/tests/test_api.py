
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@pytest.fixture()
def request_below_50k():
    """label <=50K from data"""
    request = {
        "age":53,
        "workclass":"Private",
        "fnlgt":127671,
        "education":"7th-8th",
        "education_num":4,
        "marital_status":"Married-civ-spouse",
        "occupation":"Machine-op-inspct",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital_gain":0,
        "capital_loss":0,
        "hours_per_week":40,
        "native_country":"United-States"
    }
    return request


@pytest.fixture()
def request_above_50k():
    """label >50K from data"""
    request = {
        "age":42,
        "workclass":"Private",
        "fnlgt":159449,
        "education":"Masters",
        "education_num":14,
        "marital_status":"Married-civ-spouse",
        "occupation":"Exec-managerial",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital_gain":5178,
        "capital_loss":0,
        "hours_per_week":40,
        "native_country":"United-States"
    }
    return request


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the project!"}

def test_predict_below_50k(request_below_50k):
    response = client.post("/inference", json=request_below_50k)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_predict_above_50k(request_above_50k):
    response = client.post("/inference", json=request_above_50k)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}

