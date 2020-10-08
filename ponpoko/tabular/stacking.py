from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, has_fit_parameter
from sklearn.model_selection import KFold, StratifiedKFold

class StackingTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y=y, **fit_params)

    def transform(self, X, y, sample_weight=None):
        pass