from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """
    The 'MetaEnum' class is a metaclass for creating custom Enum classes. It extends the
    'EnumMeta' metaclass provided by the 'enum' module.

    Methods: - __contains__(cls, item): This method is used to check if an item is a valid
    member of the Enum class. It tries to create an instance of the Enum class with the given
    item and if it raises a ValueError, it means the item is not a valid member. Returns True if
    the item is a valid member, otherwise False.

    Note: This class should not be used directly, but rather as a metaclass for creating custom
    Enum classes.
    """

    def __contains__(cls, item):
        """
        Check if an item is a valid member of the Enum class.

        Parameters
        ----------
        - cls: The Enum class.
        - item: The item to check.

        Returns
        -------
        - True if the item is a valid member of the Enum class, otherwise False.

        Note: This method tries to create an instance of the Enum class with the given item. If
        it raises a ValueError, it means the item is not a valid member.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    """
    The 'BaseEnum' class is a custom Enum class that extends the 'Enum' class provided by the
    'enum' module. It uses the 'MetaEnum' metaclass to add additional functionality to the Enum
    class.

    Methods
    -------
    - list(cls): This class method returns a list of all the values of the Enum class.

    Note: This class should be used as a base class for creating custom Enum classes.
    """

    @classmethod
    def list(cls):
        """
        Return a list of all the values of the Enum class.

        Parameters
        ----------
        - cls: The Enum class.

        Returns
        -------
        - A list of all the values of the Enum class.

        Note:
        This method should be called on the Enum class itself, not on an instance of the class.
        """
        return list(map(lambda c: c.value, cls))
