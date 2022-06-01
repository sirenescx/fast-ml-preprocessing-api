from pandas import Column


def is_zero_variance(column: Column):
    return column.nunique() == 1
