#HW 5
##Objective: Implement a full end2end MLOps using wide and deep DNN model with proper feature engineering and preprocessing on the NYC Taxi Fare dataset from Kaggle.

Build with:

Tensorflow (EDA/VSA included) (1)
Tensorflow with TFX (2)
Pytorch (3)
XGBoost (4)
Feature Engineering:

Trip Distance
division of pickup_datetime to HOUR,DAY, YEAR and MONTH
Techniques applied:

One-Hot encode all categorical attributes
Feature cross - Cross-join categorical attributes (ex: Lat cross Lon)
Network architecture - Wide (Categorical) and Deep (Continuous features)
Regularization(Overfit prevention) - L2
Uses GPU/TPU as distribution infrastructure
Highest Testing Accuracy Achieved:

Tensorflow (Train RMSE = 5.6225, Val RMSE = 6.2322):
10kk training data, 3 Hidden Layer, MSE Loss, ReLU Activation, Adam Optimizer, Learning Rate = 1e-05, Number of Epoch = 50
Tensorflow with TFX (Train RMSE = 5.8757, Testing RMSE = 3.7620):
10k training data, 3 Hidden Layer, MSE Loss, ReLU Activation, Adam Optimizer, Learning Rate = 1e-05, Number of Epoch = 50
Pytorch (Train RMSE = 5.5701 , Testing RMSE = 4.8073):
10k training data, MSE Loss, ReLU Activation, Adam Optimizer, Learning Rate = 1e-2, Number of Epoch = 100
XGBoost (Train RMSE = 2.5046, Testing RMSE = 3.8093):
10k training data, Early stopping, Max Depth = 8, Gamma = 0, ETA = 0.05
Reference/Modified code from the following notebooks:

https://colab.research.google.com/drive/1xS2YjhCYGnOrsyVRKxqN-PvdFROSnDvM?authuser=1#scrollTo=fK9LdIbXN87L
https://colab.research.google.com/gist/rafiqhasan/2164304ede002f4a8bfe56e5434e1a34/dl-e2e-taxi-dataset-tfx-e2e.ipynb
