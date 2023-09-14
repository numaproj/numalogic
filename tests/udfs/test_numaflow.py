import unittest
from datetime import datetime

import numpy.typing as npt
from pynumaflow.function import Datum, Messages, Message
from pynumaflow.function._dtypes import DatumMetadata

from numalogic.tools.types import artifact_t
from numalogic.udfs import NumalogicUDF


class DummyUDF(NumalogicUDF):
    def __init__(self):
        super().__init__()

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        val = datum.value
        return Messages(Message(value=val, keys=keys))

    @classmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **kwargs):
        pass


class DummyAsyncUDF(NumalogicUDF):
    def __init__(self):
        super().__init__(is_async=True)

    async def aexec(self, keys: list[str], datum: Datum) -> Messages:
        val = datum.value
        return Messages(Message(value=val, keys=keys))

    @classmethod
    def compute(cls, model: artifact_t, input_: npt.NDArray[float], **kwargs):
        pass


class TestNumalogicUDF(unittest.TestCase):
    def setUp(self) -> None:
        self.datum = Datum(
            keys=["k1", "k2"],
            value=b"100",
            event_time=datetime.now(),
            watermark=datetime.now(),
            metadata=DatumMetadata("1", 0),
        )

    def test_exec(self):
        udf = DummyUDF()
        msgs = udf.exec(["key1", "key2"], self.datum)
        self.assertIsInstance(msgs, Messages)

    def test_call(self):
        udf = DummyUDF()
        msgs = udf(["key1", "key2"], self.datum)
        self.assertIsInstance(msgs, Messages)

    async def test_aexec(self):
        udf = DummyUDF()
        with self.assertRaises(NotImplementedError):
            await udf.aexec(["key1", "key2"], self.datum)


class TestNumalogicAsyncUDF(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.datum = Datum(
            keys=["k1", "k2"],
            value=b"100",
            event_time=datetime.now(),
            watermark=datetime.now(),
            metadata=DatumMetadata("1", 0),
        )

    async def test_aexec(self):
        udf = DummyAsyncUDF()
        msgs = await udf.aexec(["key1", "key2"], self.datum)
        self.assertIsInstance(msgs, Messages)

    async def test_call(self):
        udf = DummyAsyncUDF()
        msgs = await udf(["key1", "key2"], self.datum)
        self.assertIsInstance(msgs, Messages)

    def test_exec(self):
        udf = DummyAsyncUDF()
        with self.assertRaises(NotImplementedError):
            udf.exec(["key1", "key2"], self.datum)


if __name__ == "__main__":
    unittest.main()
