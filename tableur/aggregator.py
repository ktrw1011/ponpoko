from typing import List, Dict
import pandas as pd
import numpy as np

def generate_statics_features(df: pd.DataFrame, pk: List[str], agg_funcs, suffix=""):
    """
    simple groupby

    examples
    ========
    |    | city   | cat   |   target |
    |---:|:-------|:------|---------:|
    |  0 | tokyo  | A     |        0 |
    |  1 | nagoya | B     |        1 |
    |  2 | osaka  | A     |        0 |
    |  3 | tokyo  | B     |        1 |
    |  4 | nagoya | A     |        0 |
    |  5 | osaka  | C     |        1 |
    |  6 | tokyo  | A     |        0 |
    |  7 | osaka  | C     |        1 |
    |  8 | tokyo  | A     |        0 |

    aggregator.generate_statics_features(df, ["city"], {"target":["count"]})

    |    | city   |   cat_count |
    |---:|:-------|------------:|
    |  0 | nagoya |           2 |
    |  1 | osaka  |           3 |
    |  2 | tokyo  |           4 |
    """
    
    agg_pvs = df.groupby(pk).agg(agg_funcs)
        
    rename_columns = ['_'.join(col).strip() for col in agg_pvs.columns.values]
    
    if suffix != "":
        rename_columns = [suffix+col for col in rename_columns]
        
    agg_pvs.columns = rename_columns
    
    agg_pvs.reset_index(inplace=True)
    
    return agg_pvs


def pivot_agg(df, pk:List[str], agg_funcs, pivot_col: str, pivot_index: List[str]=None, suffix=""):
    """
    categoryで集計して、categoryをone-hot-encodingするイメージ

    examples
    ========
    aggregator.generate_statics_features(df, ["city", "cat"], {"target":[sum]})

    |    | city   | cat   |   target_sum |
    |---:|:-------|:------|-------------:|
    |  0 | nagoya | A     |            0 |
    |  1 | nagoya | B     |            1 |
    |  2 | osaka  | A     |            0 |
    |  3 | osaka  | C     |            2 |
    |  4 | tokyo  | A     |            0 |
    |  5 | tokyo  | B     |            1 |

    aggregator.pivot_agg(df, ["city", "cat"], {"target":[sum]}, pivot_index=["city"], pivot_cols="cat")

    |    | city   |   cat_A_target_sum |   cat_B_target_sum |   cat_C_target_sum |
    |---:|:-------|-------------------:|-------------------:|-------------------:|
    |  0 | nagoya |                  0 |                  1 |                nan |
    |  1 | osaka  |                  0 |                nan |                  2 |
    |  2 | tokyo  |                  0 |                  1 |                nan |
    """
    temp = generate_statics_features(df, pk, agg_funcs, suffix=suffix)
    
    if pivot_index is None:
        pivot_index = pk

    temp = temp.pivot_table(index=pivot_index, columns=pivot_col)
    
    col_names = []
    for col in temp.columns:
        col_names.append(pivot_col + "_"+str(col[1])+"_"+str(col[0]))
        
    temp.columns = col_names
    
    temp = temp.reset_index()
    
    return temp