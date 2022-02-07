import xlwings as xw
import pandas as pd

@xw.func
@xw.arg('x', pd.DataFrame)
def correl5(x):
    # x arrives as DataFrame
    return x.corr()