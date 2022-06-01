import os

import pandas as pd

from fast_ml_preprocessing.services.pipeline.settings import DataFrameSettings
from fast_ml_preprocessing.utils.filesystem_utils import get_file_extension


class DatasetWritingOperation:
    def write(self, features: pd.DataFrame, target: pd.Series, filepath: str) -> str:
        features[DataFrameSettings.target_column_name] = target.values
        os.remove(filepath)
        filepath = filepath.replace(get_file_extension(filepath), ".csv")
        features.to_csv(filepath)
        return filepath

    def write_prediction(self, features: pd.DataFrame, filepath: str) -> str:
        os.remove(filepath)
        filepath = filepath.replace(get_file_extension(filepath), ".csv")
        features.to_csv(filepath)
        return filepath
