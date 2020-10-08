import numbers
import numpy as np
import pandas as pd
from typing import List, Optional, Iterable, Union

import sklearn.model_selection as model_selection
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target

def check_cv(
    cv: Union[int, Iterable, BaseCrossValidator]=5,
    y: Optional[Union[pd.Series, np.ndarray]]=None,
    stratified: bool=False,
    random_state: int=0,
    ):
    """[summary]

    Args:
        cv (Union[int, Iterable, BaseCrossValidator], optional): [description]. Defaults to 5.
        y (Optional[Union[pd.Series, np.ndarray]], optional): [description]. Defaults to None.
        stratified (bool, optional): [description]. Defaults to False.
        random_state (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]

    Example:
        check_cv(5, binary or multiclass target, stratified=True)
        >>> StratifiedKFold
        check_cv(StratifiedKFold, binary or multiclass target)
        >>> sklearn.model_selection._split.StratifiedKFold
    """

    if cv is None:
        cv = 5
    if isinstance(cv, numbers.Integral):
        if stratified and (y is not None) and (type_of_target(y) in ('binary', 'multiclass')):
            return StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            # 連続値とかならこっち
            return KFold(cv, shuffle=True, random_state=random_state)

    return model_selection.check_cv(cv, y, stratified)