import sys
import threading

import aiorun
from pynumaflow.function import Server, AsyncServer
from pynumaflow.sink import Sink

from src._constants import CONFIG_PATHS
from src.factory import HandlerFactory
from src.watcher import Watcher, ConfigHandler


def run_watcher():
    w = Watcher(CONFIG_PATHS, ConfigHandler())
    w.run()


if __name__ == "__main__":
    background_thread = threading.Thread(target=run_watcher, args=())
    background_thread.daemon = True
    background_thread.start()

    step_handler = HandlerFactory.get_handler(sys.argv[2])
    server_type = sys.argv[1]

    if server_type == "udsink":
        server = Sink(sink_handler=step_handler)
        server.start()
    elif server_type == "udf":
        server = Server(map_handler=step_handler)
        server.start()
    elif server_type == "async_udf":
        server = AsyncServer(reduce_handler=step_handler)
        aiorun.run(server.start())
    else:
        raise ValueError(f"sys arg: {server_type} not understood!")
