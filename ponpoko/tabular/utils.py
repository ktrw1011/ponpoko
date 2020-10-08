from logging import DEBUG
import numpy as np
import pandas as pd

from lightgbm.callback import _format_eval_result


def get_cat_cols(df):
    cols = df.select_dtypes(include="object").columns
    return cols

def get_num_cols(df, specific_exclude_cols=None):
    cols = df.select_dtypes(exclude=["object", "datetime64[ns, UTC]", "bool"]).columns
    return cols

def dict_formater(cols):
    cols = sorted(cols)
    return {col_name:None for col_name in cols}

def convert_datetime(df, col_names, make_feature=True):
    df[col_names] = pd.to_datetime(df[col_names], utc=True)
    
    if make_feature:
        for col_name in col_names:
            df[col_name+"_dow"] = df[col_name].dt.dayofweek
            df[col_name+"_is_weekend"] = (df[col_name].dt.weekday >= 5).astype(int)
            df[col_name+"_hour"] = df[col_name].dt.hour
            df[col_name+"_minute"] = df[col_name].dt.minute
            df[col_name+"_second"] = df[col_name].dt.second
    
    return df

def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback