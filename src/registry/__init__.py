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


from registry import LocalLRUCache


__all__ = ["ArtifactManager", "ArtifactData", "ArtifactCache", "LocalLRUCache"]


try:
    from registry import MLflowRegistry  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("MLflowRegistry")

try:
    from registry import RedisRegistry  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("RedisRegistry")

try:
    from registry import DynamoDBRegistry  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("DynamoDBRegistry")
