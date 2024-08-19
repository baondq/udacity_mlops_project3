# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Add the necessary imports for the starter code.
from modules.ml import process_data, train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv("data/cleaned_census.csv")
print(f"number of samples = {data.shape[0]}")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
print(f"number of training samples = {train.shape[0]}")
print(f"number of test samples = {test.shape[0]}")

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
print(f"X_test shape = {X_test.shape}")
print(f"y_test shape = {y_test.shape}")

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)
with open("model/logistic_regression.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoder": encoder,
        "lb": lb
    }, f)

# Evaluate model on test data
preds = inference(model=model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"fbeta = {fbeta}")
