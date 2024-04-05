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

from enum import Enum, EnumMeta

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class MetaEnum(EnumMeta):
    """
    The 'MetaEnum' class is a metaclass for creating custom Enum classes. It extends the 'EnumMeta' metaclass
    provided by the 'enum' module.

    Methods: - __contains__(cls, item): This method is used to check if an item is a valid member of the Enum class.
    It tries to create an instance of the Enum class with the given item and if it raises a ValueError, it means the
    item is not a valid member. Returns True if the item is a valid member, otherwise False.

    Note: This class should not be used directly, but rather as a metaclass for creating custom Enum classes.
    """

    def __contains__(cls, item):
        """
        Check if an item is a valid member of the Enum class.

        Parameters:
        - cls: The Enum class.
        - item: The item to check.

        Returns:
        - True if the item is a valid member of the Enum class, otherwise False.

        Note:
        This method tries to create an instance of the Enum class with the given item. If it raises a ValueError, it means the item is not a valid member.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    """
    The 'BaseEnum' class is a custom Enum class that extends the 'Enum' class provided by the 'enum' module. It uses the 'MetaEnum' metaclass to add additional functionality to the Enum class.

    Methods:
    - list(cls): This class method returns a list of all the values of the Enum class.

    Note: This class should be used as a base class for creating custom Enum classes.
    """

    @classmethod
    def list(cls):
        """
        Return a list of all the values of the Enum class.

        Parameters:
        - cls: The Enum class.

        Returns:
        - A list of all the values of the Enum class.

        Note:
        This method should be called on the Enum class itself, not on an instance of the class.
        """
        return list(map(lambda c: c.value, cls))
