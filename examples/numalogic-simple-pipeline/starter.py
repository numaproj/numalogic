import sys

from pynumaflow.function import UserDefinedFunctionServicer

from ml_steps.udf_factory import HandlerFactory

if __name__ == "__main__":
    step_handler = HandlerFactory.get_handler(sys.argv[1])
    grpc_server = UserDefinedFunctionServicer(step_handler)
    grpc_server.start()
