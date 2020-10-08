import numbers

import numpy as np
import pandas as pd
from typing import List, Optional, Iterable, Union

import sklearn.model_selection as model_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.multiclass import type_of_target

import category_encoders as ce
from category_encoders.utils import convert_input, convert_input_vector

from .validation import check_cv

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cv: Optional[Union[int, Iterable, BaseCrossValidator]]=None,
        cols: List[str]=None,
        drop_invariant: bool = False, handle_missing: str = 'value', handle_unknown: str = 'value',
        min_samples_leaf: int = 1, smoothing: float = 1.0,
        return_same_type: bool=True,
        groups: Optional[pd.Series] = None,):

        self.cv = cv
        self.n_splits = None

        ce_params = {
            "cols":cols, "return_df":True,
            "drop_invariant":drop_invariant, "handle_missing":handle_missing,
            "handle_unknown":handle_unknown, "min_samples_leaf":min_samples_leaf,
            "smoothing":smoothing,
            }

        self.base_transformer = ce.TargetEncoder(**ce_params)
        self.transformers = None

        self.return_same_type = return_same_type
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._post_fit(self.fit_transform(X, y), y)

        return self

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: pd.Series=None, **fit_params) -> Union[pd.DataFrame, np.ndarray]:
        assert len(X) == len(y)

        self._pre_train(y)
        is_pandas = isinstance(X, pd.DataFrame)

        X = convert_input(X)
        y = convert_input_vector(y, X.index)

        if y.isnull().sum() > 0:
            # 欠損値が存在
            # y == null is regarded as test data
            X_ = X.copy()
            X_.loc[~y.isnull(), :] = self._fit_train(X[~y.isnull()], y[~y.isnull()], **fit_params)
            X_.loc[y.isnull(), :] = self._fit_train(X[y.isnull()], None, **fit_params)
        else:
            X_ = self._fit_train(X, y, **fit_params)

        X_ = self._post_transform(self._post_fit(X_, y))

        return X_ if self.return_same_type and is_pandas else X_.values


    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        is_pandas = isinstance(X, pd.DataFrame)
        X_ = self._fit_train(X, None)
        X_ = self._post_transform(X_)

        return X_ if self.return_same_type and is_pandas else X_.values

    def _pre_train(self, y):
        #cv確認
        self.cv = check_cv(self.cv, y)
        self.n_splits = self.cv.get_n_splits()
        
        # category encoderをクローン
        # +1はtrain全体でfitさせて、testをtransformする用
        self.transformers = [clone(self.base_transformer) for _ in range(self.n_splits + 1)]

    def _fit_train(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params) -> pd.DataFrame:
        # fitの実体

        if y is None:
            X_ = self.transformers[-1].transform(X)
            return self._post_transform(X_)

        X_ = X.copy()

        # trainに対してはfoldでcategory encode
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(X_, y, self.groups)):
            self.transformers[i].fit(X.iloc[trn_idx], y.iloc[trn_idx], **fit_params)
            X_.iloc[val_idx, :] = self.transformers[i].transform(X.iloc[val_idx])

        # テスト用にに全体でcategory encode
        self.transformers[-1].fit(X, y, **fit_params)

        return X_

    def _post_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X
    
    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.transformers[0].cols
        for c in cols:
            X[c] = X[c].astype(float)
        return X
