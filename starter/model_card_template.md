# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistic Regression were trained.

* Model version: 1.0.0
* Model date: 18 August 2024

## Intended Use
The model can be used for predicting income classes on census data. There are two income classes >50K and <=50K (binary classification task).

## Training Data
The UCI Census Income Data Set was used for training. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
For training 80% of the 30162 rows were used (24129 rows) in the training set.

## Evaluation Data
For evaluation 20% of the 30162 rows were used (6033 instances) in the test set.

## Metrics
Three metrics were used for model evaluation (performance on test set):
* precision: 0.6189873417721519
* recall: 0.31877444589308995
* fbeta: 0.4208261617900172

## Ethical Considerations
Since the dataset consists of public available data with highly aggregated census data no harmful unintended use of the data has to be addressed.

## Caveats and Recommendations
It would be meaningful to perform an hyperparameter optimization to improve the model performance.