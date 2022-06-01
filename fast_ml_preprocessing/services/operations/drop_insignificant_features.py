import pandas as pd


class InsignificantFeaturesDroppingOperation:
    def drop(self, features: pd.DataFrame) -> pd.DataFrame:
        dataset_length: int = len(features)

        categorical_columns: list[str] = list(
            features.select_dtypes(exclude=[pd.Int64Dtype(), pd.Float64Dtype(), pd.BooleanDtype()]).columns
        )
        for column in categorical_columns:
            if features[column].nunique() == dataset_length or features[column].nunique() == 1:
                features = features.drop(column, axis=1)

        columns: list[str] = features.columns.tolist()
        for column in columns:
            if features[column].nunique() == 1:
                features = features.drop(column, axis=1)

        return features

    # def drop_prediction(self, features: pd.DataFrame, filepath: str) -> pd.DataFrame:
    #
    # def _get_features_for_encoding(self, filepath: str) -> (pd.DataFrame, pd.Series):
    #     data: pd.DataFrame = pd.read_csv(filepath, sep=',')
    #     features: pd.DataFrame = data.drop(DataFrameSettings.target_column_name, axis=1)
    #     target: pd.Series = data[DataFrameSettings.target_column_name]
    #     return features, target
