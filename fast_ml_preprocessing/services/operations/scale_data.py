import pandas as pd

from sklearn.preprocessing import StandardScaler


class DataScalingOperation:
    def scale(self, features: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.values)
        features = pd.DataFrame(scaled_features, columns=features.columns)

        return features



