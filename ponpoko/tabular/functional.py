from typing import List, Union, Optional, Iterable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator
from sklearn.utils import check_X_y, check_array

def stacking(
    models: List,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:Optional[np.ndarray],
    sample_weights: np.ndarray,
    cv: Optional[Union[int, Iterable, BaseCrossValidator]],
    mode:str):
    if len(models) == 0:
        raise ValueError("List of models is empty")

    if X_test is None and mode != "oof":
        raise ValueError("X_test can be None only if mode='oof'")

    # check arrays
    # y_train and sample weights must be 1d ndarray (sample_size, )
    X_train, y_train = check_X_y(
        X_train,
        y_train,
        accept_sparse=["csr"],
        force_all_finite=False, # allow nan and inf. some models(xgboost) can be handle
        allow_nd=True,
        multi_output=False,
    )

    if X_test is not None:
        X_test = check_array(
            X_test,
            accept_sparse=["csr"],
            allow_nd=True,
            force_all_finite=False,
        )

    if sample_weights is not None:
        sample_weights = np.array(sample_weights).ravel()

    if mode not in ['pred', 'pred_bag', 'oof', 'oof_pred']:
        raise ValueError