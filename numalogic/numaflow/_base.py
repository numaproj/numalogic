from pynumaflow.function import Datum, Messages


class NumalogicUDF:
    """
    Base class for all Numalogic based UDFs.

    Args:
        is_async: If True, the UDF is executed in an asynchronous manner.
    """

    __slots__ = ("is_async",)

    def __init__(self, is_async=False):
        self.is_async = is_async

    def __call__(self, keys: list[str], datum: Datum) -> Messages:
        if self.is_async:
            return self.aexec(keys, datum)
        return self.exec(keys, datum)

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Called when the UDF is executed in a synchronous manner.

        Args:
            keys: list of keys.
            datum: Datum object.

        Returns
        -------
            Messages instance
        """
        raise NotImplementedError("exec method not implemented")

    async def aexec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Called when the UDF is executed in an asynchronous manner.

        Args:
            keys: list of keys.
            datum: Datum object.

        Returns
        -------
            Messages instance
        """
        raise NotImplementedError("aexec method not implemented")
