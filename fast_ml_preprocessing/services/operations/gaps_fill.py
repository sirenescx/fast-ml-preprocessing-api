import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn_pandas import CategoricalImputer

from fast_ml_preprocessing.services.pipeline.settings import DataFrameSettings
from fast_ml_preprocessing.utils.problem_type import ProblemType


class MissingValuesFillingOperation:
    def fill_missing_values_in_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = self._convert_integer_columns_to_int_64(features)
        features = self._remove_columns_with_half_or_more_missing_values(features)

        columns_with_gaps: set[str] = self._get_columns_with_gaps(features)
        continuous_columns: list[str] = features.select_dtypes(include=pd.Float64Dtype()).columns.tolist()
        boolean_and_integer_columns: list[str] = \
            features.select_dtypes(include=[pd.Int64Dtype(), pd.BooleanDtype()]).columns.tolist()
        non_categorical_columns: set[str] = set(continuous_columns + boolean_and_integer_columns)
        categorical_columns: list[str] = list(set(features.columns.tolist()).difference(non_categorical_columns))

        features = self._fill_gaps_in_continuous_columns(
            features,
            continuous_columns,
            columns_with_gaps
        )
        features = self._fill_gaps_in_boolean_and_integer_columns(
            features,
            boolean_and_integer_columns,
            columns_with_gaps
        )
        features = self._fill_gaps_in_categorical_columns(
            features,
            categorical_columns,
            columns_with_gaps
        )
        return features

    def fill_missing_values_in_target(self, target: pd.Series, problem_type: ProblemType, index: pd.Index) -> pd.Series:
        if problem_type == ProblemType.REGRESSION.value:
            if target.isna().values.any():
                imputer: KNNImputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
                target_df = target.to_frame(DataFrameSettings.target_column_name)
                imputed_target = pd.DataFrame(
                    imputer.fit_transform(target_df[[DataFrameSettings.target_column_name]]),
                    columns=[target_df.columns],
                    index=index
                )
                return imputed_target[DataFrameSettings.target_column_name]
            else:
                return target

        if target.isna().values.any():
            try:
                if np.array_equal(target.values, target.values.astype(int)):
                    target = target.astype(pd.Int64Dtype())

                target_df = target.to_frame(DataFrameSettings.target_column_name)
                imputer: SimpleImputer = SimpleImputer(strategy='most_frequent')
                imputed_target = pd.DataFrame(
                    imputer.fit_transform(target_df[[DataFrameSettings.target_column_name]]),
                    columns=[target_df.columns],
                    index=index
                )
                return imputed_target[DataFrameSettings.target_column_name]
            except:
                imputer: CategoricalImputer = CategoricalImputer()
                target.update(
                    pd.DataFrame(
                        imputer.fit_transform(target),
                        columns=target,
                        index=index,
                    )
                )
                return target

        return target

    # set columns with integer values Int64 type
    def _convert_integer_columns_to_int_64(self, features: pd.DataFrame) -> pd.DataFrame:
        columns: list[str] = features.columns.tolist()
        for column in columns:
            non_null_values = features[features[column].notnull()][column]
            try:
                if np.array_equal(non_null_values, non_null_values.astype(int)):
                    features[column] = features[column].astype(pd.Int64Dtype())
            except:
                continue
        return features

    # remove columns with > 50% of missing values
    def _remove_columns_with_half_or_more_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        columns_to_remove: list[str] = features.columns[features.isna().sum() > len(features) / 2]
        for column in columns_to_remove:
            features.drop(column, axis=1, inplace=True)
        return features

    def _get_columns_with_gaps(self, features: pd.DataFrame) -> set[str]:
        return set(features.columns[features.isna().any()].tolist())

    def _fill_gaps_in_continuous_columns(
            self,
            features: pd.DataFrame,
            continuous_columns: list[str],
            columns_with_gaps: set[str]
    ) -> pd.DataFrame:
        columns: set[str] = set(continuous_columns)
        columns_with_gaps: list[str] = list(columns_with_gaps.intersection(columns))
        if len(columns_with_gaps) > 0:
            imputer: KNNImputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            for column in columns_with_gaps:
                features.update(
                    pd.DataFrame(
                        imputer.fit_transform(features[[column]]),
                        columns=[column],
                        index=features.index,
                        dtype=features[column].dtypes
                    )
                )
        return features

    def _fill_gaps_in_boolean_and_integer_columns(
            self,
            features: pd.DataFrame,
            boolean_and_integer_columns: list[str],
            columns_with_gaps: set[str]
    ) -> pd.DataFrame:
        columns: set[str] = set(boolean_and_integer_columns)
        columns_with_gaps: list[str] = list(columns_with_gaps.intersection(columns))
        if len(columns_with_gaps) > 0:
            imputer: SimpleImputer = SimpleImputer(strategy='most_frequent')
            for column in columns_with_gaps:
                features.update(
                    pd.DataFrame(
                        imputer.fit_transform(features[column]),
                        columns=[column],
                        index=features.index,
                        dtype=features[column].dtypes
                    )
                )
        return features

    def _fill_gaps_in_categorical_columns(
            self,
            features: pd.DataFrame,
            categorical_columns: list[str],
            columns_with_gaps: set[str]
    ) -> pd.DataFrame:
        columns: set[str] = set(categorical_columns)
        columns_with_gaps: list[str] = list(columns_with_gaps.intersection(columns))
        features[columns_with_gaps] = features[columns_with_gaps].convert_dtypes()
        if len(columns_with_gaps) > 0:
            imputer: CategoricalImputer = CategoricalImputer()
            for column in columns_with_gaps:
                features.update(
                    pd.DataFrame(
                        imputer.fit_transform(features[column]),
                        columns=[column],
                        index=features.index,
                        dtype=features[column].dtypes
                    )
                )
        return features
