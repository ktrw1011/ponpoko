import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_score_fn(y_targets, y_preds):
    y_preds = np.argmax(y_preds, axis=1)
    return accuracy_score(y_targets, y_preds)