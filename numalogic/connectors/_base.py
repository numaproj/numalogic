from abc import ABCMeta, abstractmethod

import pandas as pd


class DataFetcher(metaclass=ABCMeta):
    __slots__ = ("url",)

    def __init__(self, url: str):
        self.url = url

    @abstractmethod
    def fetch_data(self, *args, **kwargs) -> pd.DataFrame:
        pass


class AsyncDataFetcher(metaclass=ABCMeta):
    __slots__ = ("url",)

    def __init__(self, url: str):
        self.url = url

    @abstractmethod
    async def fetch_data(self, *args, **kwargs) -> pd.DataFrame:
        pass
