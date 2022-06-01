import os.path

import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
from word2number import w2n

from fast_ml_preprocessing.services.pipeline.settings import DataFrameSettings


class FeatureEncodingOperation:
    def encode(self, features: pd.DataFrame, target: pd.Series, filepath: str) -> pd.DataFrame:
        categorical_columns = (
            features
            .select_dtypes(exclude=[pd.Int64Dtype(), pd.Float64Dtype(), pd.BooleanDtype()]).columns.tolist()
        )

        for column in categorical_columns:
            try:
                features[column] = features[column].apply(w2n.word_to_num)
            except ValueError:
                continue

        binary_columns = features[filter(lambda c: features[c].nunique() == 2, categorical_columns)].columns

        if categorical_columns:
            if categorical_columns.__contains__(binary_columns):
                categorical_columns.remove(binary_columns)

        label_encoder: LabelEncoder = LabelEncoder()
        for column in binary_columns:
            label_encoder.fit(features[column])
            features[column] = label_encoder.transform(features[column])

        self._save_features_for_encoding(features[categorical_columns], target, filepath)
        cat_boost_encoder: CatBoostEncoder = CatBoostEncoder(cols=categorical_columns, drop_invariant=True)
        features = cat_boost_encoder.fit_transform(features, target)

        return features

    def encode_prediction(self, features: pd.DataFrame, filepath: str) -> pd.DataFrame:
        categorical_columns = (
            features
            .select_dtypes(exclude=[pd.Int64Dtype(), pd.Float64Dtype(), pd.BooleanDtype()]).columns.tolist()
        )

        for column in categorical_columns:
            try:
                features[column] = features[column].apply(w2n.word_to_num)
            except ValueError:
                continue

        binary_columns = features[filter(lambda c: features[c].nunique() == 2, categorical_columns)].columns

        if categorical_columns:
            if categorical_columns.__contains__(binary_columns):
                categorical_columns.remove(binary_columns)

        label_encoder: LabelEncoder = LabelEncoder()
        for column in binary_columns:
            label_encoder.fit(features[column])
            features[column] = label_encoder.transform(features[column])

        train_features, target = self._get_features_for_encoding(filepath=filepath)
        cat_boost_encoder: CatBoostEncoder = CatBoostEncoder(cols=categorical_columns, drop_invariant=True)
        cat_boost_encoder.fit(train_features, target)
        features[categorical_columns] = cat_boost_encoder.transform(features[categorical_columns])

        return features

    def _save_features_for_encoding(self, features: pd.DataFrame, target: pd.Series, filepath: str):
        path, _ = os.path.split(filepath)
        dataset_for_encoding: pd.DataFrame = pd.DataFrame(features)
        dataset_for_encoding[DataFrameSettings.target_column_name] = target
        dataset_for_encoding.to_csv(os.path.join(path, "dataset_for_encoding.csv"), sep=',')

    def _get_features_for_encoding(self, filepath: str) -> (pd.DataFrame, pd.Series):
        path, _ = os.path.split(filepath)
        data: pd.DataFrame = pd.read_csv(os.path.join(path, "dataset_for_encoding.csv"), sep=',', index_col=0)
        features: pd.DataFrame = data.drop(DataFrameSettings.target_column_name, axis=1)
        target: pd.Series = data[DataFrameSettings.target_column_name]
        return features, target
