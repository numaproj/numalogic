# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from prometheus_client import start_http_server

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def start_metrics_server(port: int) -> None:
    """
    Starts the Prometheus monitoring server.

    Args:
        port: Port number
    """
    _LOGGER.info("Starting Prometheus monitoring server on port: %s", port)
    start_http_server(port)
