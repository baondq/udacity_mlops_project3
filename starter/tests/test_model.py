
import pandas as pd
import pandas.api.types as pdtypes
import pytest
from sklearn.model_selection import train_test_split

from modules.ml import inference, compute_model_metrics, train_model, process_data, compute_metrics_with_slices_data


fake_categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture()
def data():
    df = pd.read_csv("data/cleaned_census.csv")
    return df


@pytest.fixture()
def sliced_data(data):
    train_df, test_df = train_test_split(data, test_size=0.25, random_state=0)
    return train_df, test_df


@pytest.fixture()
def processed_data(sliced_data):
    train_df, test_df = sliced_data
    X_train, y_train, encoder, lb = process_data(
        X=train_df,
        categorical_features=fake_categorical_features,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=fake_categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_train, y_train, X_test, y_test, encoder, lb

@pytest.fixture()
def model(processed_data):
    X_train, y_train, _, _, _, _ = processed_data
    model = train_model(
        X_train=X_train,
        y_train=y_train
    )
    return model


def test_column_presence_and_type(data):
    """Tests that cleaned csv file has expected columns and types.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }
    assert set(data.columns.values).issuperset(set(required_columns.keys()))
    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_workclass_values(data):
    """Tests that the workclass column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    }
    workclass_list = list(data["workclass"].unique())
    for wc in workclass_list:
        assert wc in expected_values, f"{wc} not in expected values"


def test_education_values(data):
    """Tests that the education column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    }
    education_list = list(data["education"].unique())
    for e in education_list:
        assert e in expected_values, f"{e} not in expected values"


def test_marital_status_values(data):
    """Tests that the marital-status column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }
    marital_status_list = list(data["marital-status"].unique())
    for ms in marital_status_list:
        assert ms in expected_values, f"{ms} not in expected values"


def test_occupation_values(data):
    """Tests that the occupation column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    }
    occupation_list = list(data["occupation"].unique())
    for o in occupation_list:
        assert o in expected_values, f"{o} not in expected values"


def test_relationship_values(data):
    """Tests that the relationship column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    }
    relationship_list = list(data["relationship"].unique())
    for r in relationship_list:
        assert r in expected_values, f"{r} not in expected values"


def test_sex_values(data):
    """Tests that the sex column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Male",
        "Female"
    }
    sex_list = list(data["sex"].unique())
    for s in sex_list:
        assert s in expected_values, f"{s} not in expected values"


def test_salary_values(data):
    """Tests that the salary column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "<=50K",
        ">50K"
    }
    salary_list = list(data["salary"].unique())
    for s in salary_list:
        assert s in expected_values, f"{s} not in expected values"


def test_column_ranges(data):

    ranges = {
        "age": (17, 90),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }
    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].min() >= minimum
        assert data[col_name].max() <= maximum


def test_column_values(data):
    # Check that the columns are of the right dtype
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Column {col_name} has null values"


def test_model_input(data):
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Features {col_name} has null values"


def test_inference(processed_data, model):
    """
    Assert that inference function returns correct
    amount of predictions with respect to the input
    """
    _, _, X_test, y_test, _, _ = processed_data
    preds = inference(model, X_test)
    assert preds.shape[0] == y_test.shape[0]


def test_compute_model_metrics(processed_data, model):
    """
    Assert that output metrics are in the correct range
    """
    _, _, X_test, y_test, _, _ = processed_data
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

    assert precision >= 0.0 and precision <= 1.0
    assert recall >= 0.0 and recall <= 1.0
    assert fbeta >= 0.0 and fbeta <= 1.0


def test_compute_metrics_with_slices_data(data, processed_data, model):
    df, _ = train_test_split(data, train_size=0.5)
    _, _, _, _, encoder, lb = processed_data
    compute_metrics_with_slices_data(
        df=df,
        cat_columns=fake_categorical_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
        slice_output_path="tests/test_data/test_compute_metrics_with_slices_data.csv"
    )

