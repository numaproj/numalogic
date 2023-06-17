import sys

from pynumaflow.function import Server
from src import Inference, Train


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Please provide a step name")

    step = sys.argv[1]
    if step == "inference":
        step_handler = Inference()
    elif step == "train":
        step_handler = Train()
    else:
        raise ValueError(f"Invalid step provided: {step}")

    grpc_server = Server(map_handler=step_handler)
    grpc_server.start()
