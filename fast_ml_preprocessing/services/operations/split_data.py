import pandas as pd


class DatasetSplittingOperation:
    def split(self, dataframe: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
        features = dataframe.drop(target, axis=1)
        target = dataframe[target]
        return features, target
