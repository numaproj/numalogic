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

import unittest
from unittest.mock import patch

import fakeredis

from numalogic.config import RegistryInfo
from numalogic.tools.exceptions import UnknownConfigArgsError


class TestOptionalDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.regconf = RegistryInfo(name="RedisRegistry", conf=dict(ttl=50))

    @patch("numalogic.config.factory.getattr", side_effect=AttributeError)
    def test_not_installed_dep_01(self, _):
        from numalogic.config.factory import RegistryFactory

        model_factory = RegistryFactory()
        server = fakeredis.FakeServer()
        redis_cli = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
        with self.assertRaises(ImportError):
            model_factory.get_cls("RedisRegistry")(redis_cli, **self.regconf.conf)

    @patch("numalogic.config.factory.getattr", side_effect=AttributeError)
    def test_not_installed_dep_02(self, _):
        from numalogic.config.factory import RegistryFactory

        model_factory = RegistryFactory()
        server = fakeredis.FakeServer()
        redis_cli = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
        with self.assertRaises(ImportError):
            model_factory.get_instance(self.regconf)(redis_cli, **self.regconf.conf)

    def test_unknown_registry(self):
        from numalogic.config.factory import RegistryFactory

        model_factory = RegistryFactory()
        reg_conf = RegistryInfo(name="UnknownRegistry")
        with self.assertRaises(UnknownConfigArgsError):
            model_factory.get_cls("UnknownRegistry")
        with self.assertRaises(UnknownConfigArgsError):
            model_factory.get_instance(reg_conf)


if __name__ == "__main__":
    unittest.main()
