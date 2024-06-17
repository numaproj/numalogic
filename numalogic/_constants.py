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


import os

NUMALOGIC_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.split(NUMALOGIC_DIR)[0]
TESTS_DIR = os.path.join(NUMALOGIC_DIR, "../tests")
BASE_CONF_DIR = os.path.join(BASE_DIR, "config")

DEFAULT_BASE_CONF_PATH = os.path.join(BASE_CONF_DIR, "default-configs", "config.yaml")
DEFAULT_METRICS_CONF_PATH = os.path.join(
    BASE_CONF_DIR, "default-configs", "numalogic_udf_metrics.yaml"
)
DEFAULT_APP_CONF_PATH = os.path.join(BASE_CONF_DIR, "app-configs", "config.yaml")
DEFAULT_METRICS_PORT = 8490
NUMALOGIC_METRICS = "numalogic_metrics"
