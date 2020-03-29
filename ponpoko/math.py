import numpy as np
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))