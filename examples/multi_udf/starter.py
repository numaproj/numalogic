import sys

from pynumaflow.function import Server

from src.factory import UDFFactory

if __name__ == "__main__":
    step_handler = UDFFactory.get_handler(sys.argv[1])
    grpc_server = Server(map_handler=step_handler)
    grpc_server.start()
