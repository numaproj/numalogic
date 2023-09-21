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


class ModelInitializationError(RuntimeError):
    """Raised when a model is not initialized properly."""

    pass


class InvalidRangeParameter(Exception):
    """Raised when the range parameter is not valid."""

    pass


class LayerSizeMismatchError(Exception):
    """Raised when the layer size is not valid."""

    pass


class DataModuleError(Exception):
    """Base class for all exceptions raised by the DataModule class."""

    pass


class InvalidDataShapeError(Exception):
    """Raised when the data shape is not valid."""

    pass


class UnknownConfigArgsError(Exception):
    """Raised when an unknown config argument is passed to a model."""

    pass


class ConfigNotFoundError(RuntimeError):
    """Raised when a config is not found."""

    pass


class ModelVersionError(Exception):
    """Raised when a model version is not found in the registry."""

    pass


class RedisRegistryError(Exception):
    """Base class for all exceptions raised by the RedisRegistry class."""

    pass


class DynamoDBRegistryError(Exception):
    """Base class for all exceptions raised by the DynamoDBRegistry class."""

    pass


class ModelKeyNotFound(RedisRegistryError):
    """Raised when a model key is not found in the registry."""

    pass


class EnvVarNotFoundError(LookupError):
    """Raised when an environment variable is not found."""

    pass


class PrometheusFetcherError(Exception):
    """Base class for all exceptions raised by the PrometheusFetcher class."""

    pass


class PrometheusInvalidResponseError(PrometheusFetcherError):
    """Raised when the Prometheus response is not a success."""

    pass


class DataFormatError(Exception):
    """Raised when the data format is not valid."""

    pass
