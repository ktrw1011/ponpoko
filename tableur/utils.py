import numpy as np
import pandas as pd


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