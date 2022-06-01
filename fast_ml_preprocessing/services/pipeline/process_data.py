import logging
import os.path

import pandas as pd

from fast_ml_preprocessing.services.operations.define_problem import ProblemTypeDefiningOperation
from fast_ml_preprocessing.services.operations.drop_insignificant_features import InsignificantFeaturesDroppingOperation
from fast_ml_preprocessing.services.operations.encode_features import FeatureEncodingOperation
from fast_ml_preprocessing.services.operations.gaps_fill import MissingValuesFillingOperation
from fast_ml_preprocessing.services.operations.read_data import DatasetReadingOperation
from fast_ml_preprocessing.services.operations.scale_data import DataScalingOperation
from fast_ml_preprocessing.services.operations.select_features import FeatureSelectionOperation
from fast_ml_preprocessing.services.operations.shuffle_data import DatasetShufflingOperation
from fast_ml_preprocessing.services.operations.split_data import DatasetSplittingOperation
from fast_ml_preprocessing.services.operations.write_data import DatasetWritingOperation


class TrainingDataProcessingPipeline:
    def __init__(
            self,
            dataset_reading_op: DatasetReadingOperation,
            dataset_splitting_op: DatasetSplittingOperation,
            dataset_shuffling_op: DatasetShufflingOperation,
            insignificant_features_dropping_op: InsignificantFeaturesDroppingOperation,
            problem_type_defining_op: ProblemTypeDefiningOperation,
            missing_values_filling_op: MissingValuesFillingOperation,
            feature_encoding_op: FeatureEncodingOperation,
            feature_selection_op: FeatureSelectionOperation,
            data_scaling_op: DataScalingOperation,
            dataset_writing_op: DatasetWritingOperation
    ):
        self._dataset_reading_op = dataset_reading_op
        self._dataset_splitting_op = dataset_splitting_op
        self._dataset_shuffling_op = dataset_shuffling_op
        self._insignificant_features_dropping_op = insignificant_features_dropping_op
        self._problem_type_defining_op = problem_type_defining_op
        self._missing_values_filling_op = missing_values_filling_op
        self._feature_encoding_op = feature_encoding_op
        self._feature_selection_op = feature_selection_op
        self._data_scaling_op = data_scaling_op
        self._dataset_writing_op = dataset_writing_op

    def process_dataset(
            self,
            filename: str,
            separator: str,
            index_col: int,
            target_col: str,
            problem_type: str):
        logger: logging.Logger = _get_logger(filename)
        logger.info("Started preprocessing training data")
        try:
            data: pd.DataFrame = self._dataset_reading_op.read(
                filepath=filename,
                index_col=index_col,
                separator=separator
            )
        except ValueError as error:
            logger.error(error)
            raise ValueError(error)
        features, target = self._dataset_splitting_op.split(dataframe=data, target=target_col)
        features, target = self._dataset_shuffling_op.shuffle(features=features, target=target)
        features = self._insignificant_features_dropping_op.drop(features=features)
        features = self._missing_values_filling_op.fill_missing_values_in_features(features=features)
        problem_type = self._problem_type_defining_op.define(problem_type=problem_type)
        target = self._missing_values_filling_op.fill_missing_values_in_target(
            target=target,
            problem_type=problem_type,
            index=features.index
        )
        features = self._feature_encoding_op.encode(features=features, target=target, filepath=filename)
        features = self._feature_selection_op.apply(features=features, target=target, problem_type=problem_type)
        features = self._data_scaling_op.scale(features=features)
        feature_names = features.columns.tolist()
        filepath: str = self._dataset_writing_op.write(features=features, target=target, filepath=filename)
        logger.info("Training data successfully preprocessed")
        logger.info("Running with features: " + ", ".join(feature_names))
        return filepath


class PredictionDataProcessingPipeline:
    def __init__(
            self,
            dataset_reading_op: DatasetReadingOperation,
            insignificant_features_dropping_op: InsignificantFeaturesDroppingOperation,
            missing_values_filling_op: MissingValuesFillingOperation,
            feature_encoding_op: FeatureEncodingOperation,
            data_scaling_op: DataScalingOperation,
            dataset_writing_op: DatasetWritingOperation
    ):
        self._dataset_reading_op = dataset_reading_op
        self._insignificant_features_dropping_op = insignificant_features_dropping_op
        self._missing_values_filling_op = missing_values_filling_op
        self._feature_encoding_op = feature_encoding_op
        self._data_scaling_op = data_scaling_op
        self._dataset_writing_op = dataset_writing_op

    def process_dataset(
            self,
            filename: str,
            separator: str,
            index_col: int
    ):
        logger: logging.Logger = _get_logger(filename)
        logger.info("Started preprocessing prediction data")
        try:
            features: pd.DataFrame = self._dataset_reading_op.read(
                filepath=filename,
                index_col=index_col,
                separator=separator
            )
        except ValueError as error:
            logger.error(error)
            raise ValueError(error)
        features = self._insignificant_features_dropping_op.drop(features=features)
        features = self._missing_values_filling_op.fill_missing_values_in_features(features=features)
        features = self._feature_encoding_op.encode_prediction(features=features, filepath=filename)
        feature_names = features.columns.tolist()
        features = self._data_scaling_op.scale(features=features)
        filepath: str = self._dataset_writing_op.write_prediction(features=features, filepath=filename)
        logger.info("Prediction data successfully preprocessed")
        logger.info("Running with features: " + ", ".join(feature_names))
        return filepath


def _get_logger(filename: str) -> logging.Logger:
    path, _ = os.path.split(filename)
    file_handler = logging.FileHandler(os.path.join(path, "log"), "a")
    formatter = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
