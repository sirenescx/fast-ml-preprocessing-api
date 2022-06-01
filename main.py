import grpc
from grpc_reflection.v1alpha import reflection
from concurrent import futures

from fast_ml_preprocessing.proto.compiled import preprocessor_pb2_grpc, preprocessor_pb2
from fast_ml_preprocessing.services.preprocessor import PreprocessingService


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    preprocessor_pb2_grpc.add_PreprocessingServiceServicer_to_server(
        PreprocessingService(), server
    )
    service_names = (
        preprocessor_pb2.DESCRIPTOR.services_by_name['PreprocessingService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    server.add_insecure_port('[::]:82')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
