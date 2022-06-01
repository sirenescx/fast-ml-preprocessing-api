import numpy as np
import pandas as pd


class DatasetShufflingOperation:
    def shuffle(self, features: pd.DataFrame, target: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        permutation = np.random.permutation(len(features))
        features = features.iloc[permutation].reset_index(drop=True)
        target = target.iloc[permutation].reset_index(drop=True)
        return features, target