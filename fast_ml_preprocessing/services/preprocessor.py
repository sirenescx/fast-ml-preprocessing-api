import grpc

from fast_ml_preprocessing.proto.compiled import preprocessor_pb2_grpc
from fast_ml_preprocessing.proto.compiled.preprocessor_pb2 import PreprocessingResponse
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
from fast_ml_preprocessing.services.pipeline.process_data import TrainingDataProcessingPipeline, \
    PredictionDataProcessingPipeline


# TODO: сделать кодирование таргета, если он категориальный ???


class PreprocessingService(preprocessor_pb2_grpc.PreprocessingServiceServicer):
    def PreprocessDataset(self, request, context):
        if request.mode == "train":
            try:
                pipeline = self._get_training_pipeline()
                filepath: str = pipeline.process_dataset(
                    filename=request.filename,
                    separator=request.separator,
                    index_col=request.index,
                    target_col=request.target,
                    problem_type=request.problem_type
                )
                return PreprocessingResponse(
                    message=filepath
                )
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, error)
                print(error)
        else:
            try:
                pipeline = self._get_prediction_pipeline()
                filepath: str = pipeline.process_dataset(
                    filename=request.filename,
                    separator=request.separator,
                    index_col=request.index
                )
                return PreprocessingResponse(
                    message=filepath
                )
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, error)
                print(error)

        context.abort(grpc.StatusCode.INTERNAL, "Failed to preprocess dataset")

    def _get_training_pipeline(self):
        dataset_reading_op: DatasetReadingOperation = DatasetReadingOperation()
        dataset_splitting_op: DatasetSplittingOperation = DatasetSplittingOperation()
        dataset_shuffling_op: DatasetShufflingOperation = DatasetShufflingOperation()
        insignificant_features_dropping_op: InsignificantFeaturesDroppingOperation = (
            InsignificantFeaturesDroppingOperation()
        )
        problem_type_defining_op: ProblemTypeDefiningOperation = ProblemTypeDefiningOperation()
        missing_values_filling_op: MissingValuesFillingOperation = MissingValuesFillingOperation()
        feature_encoding_op: FeatureEncodingOperation = FeatureEncodingOperation()
        feature_selection_op: FeatureSelectionOperation = FeatureSelectionOperation()
        data_scaling_op: DataScalingOperation = DataScalingOperation()
        dataset_writing_op: DatasetWritingOperation = DatasetWritingOperation()

        training_pipeline: TrainingDataProcessingPipeline = TrainingDataProcessingPipeline(
            dataset_reading_op=dataset_reading_op,
            dataset_splitting_op=dataset_splitting_op,
            dataset_shuffling_op=dataset_shuffling_op,
            insignificant_features_dropping_op=insignificant_features_dropping_op,
            problem_type_defining_op=problem_type_defining_op,
            missing_values_filling_op=missing_values_filling_op,
            feature_encoding_op=feature_encoding_op,
            feature_selection_op=feature_selection_op,
            data_scaling_op=data_scaling_op,
            dataset_writing_op=dataset_writing_op
        )

        return training_pipeline

    def _get_prediction_pipeline(self):
        dataset_reading_op: DatasetReadingOperation = DatasetReadingOperation()
        insignificant_features_dropping_op: InsignificantFeaturesDroppingOperation = (
            InsignificantFeaturesDroppingOperation()
        )
        missing_values_filling_op: MissingValuesFillingOperation = MissingValuesFillingOperation()
        feature_encoding_op: FeatureEncodingOperation = FeatureEncodingOperation()
        data_scaling_op: DataScalingOperation = DataScalingOperation()
        dataset_writing_op: DatasetWritingOperation = DatasetWritingOperation()

        prediction_pipeline: PredictionDataProcessingPipeline = PredictionDataProcessingPipeline(
            dataset_reading_op=dataset_reading_op,
            insignificant_features_dropping_op=insignificant_features_dropping_op,
            missing_values_filling_op=missing_values_filling_op,
            feature_encoding_op=feature_encoding_op,
            data_scaling_op=data_scaling_op,
            dataset_writing_op=dataset_writing_op
        )

        return prediction_pipeline
