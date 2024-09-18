import os
from datetime import datetime
from typing import Optional

import pandas as pd
from wavefront_api_client import Configuration, QueryApi, ApiClient

from numalogic.connectors._base import DataFetcher
from numalogic.tools.exceptions import WavefrontFetcherError

import logging

LOGGER = logging.getLogger(__name__)


class WavefrontFetcher(DataFetcher):
    """
    Fetches data from Wavefront.

    Args:
        url (str): Wavefront URL.
        api_token (str): Wavefront API token.

    Raises
    ------
        ValueError: If API token is not provided.
        WavefrontFetcherError: If there is an error fetching data from Wavefront.
    """

    def __init__(self, url: str, api_token: Optional[str] = None):
        super().__init__(url)
        api_token = api_token or os.getenv("WAVEFRONT_API_TOKEN")
        if not api_token:
            raise ValueError("WAVEFRONT API token is not provided")
        configuration = Configuration()
        configuration.host = url
        configuration.api_key["X-AUTH-TOKEN"] = api_token
        self.api_client = QueryApi(
            ApiClient(
                configuration,
                header_name="Authorization",
                header_value=f"Bearer {api_token}",
            )
        )

    def _call_api(
        self, query: str, start: int, end: Optional[int], granularity: str
    ) -> pd.DataFrame:
        """Calls the Wavefront API to fetch data."""
        return self.api_client.query_api(
            query, start, granularity, e=end, include_obsolete_metrics=True, use_raw_qk=True
        )

    @staticmethod
    def _format_results(res: dict) -> pd.DataFrame:
        """Validates and formats the results from the API."""
        if res.get("error_type") is not None:
            raise WavefrontFetcherError(
                f"Error fetching data from Wavefront: "
                f"{res.get('error_type')}: {res.get('error_message')}"
            )
        if res.get("timeseries") is None:
            raise WavefrontFetcherError("No timeseries data found for the query")
        dfs = []
        for ts in res["timeseries"]:
            dfs.append(pd.DataFrame(ts["data"], columns=["timestamp", "value"]))
        df = pd.concat(dfs)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df.set_index("timestamp").sort_index()

    def fetch(
        self,
        metric: str,
        start: datetime,
        filters: Optional[dict] = None,
        end: Optional[datetime] = None,
        granularity: str = "m",
    ) -> pd.DataFrame:
        """
        Fetches data from Wavefront as a single metric.

        Args:
            metric (str): Metric to fetch. Example: 'system.cpu.usage'.
                Do not include the 'ts()' function.
            start (datetime): Start time.
            filters (dict): Filters to apply to the query.
            end (datetime): End time. Set to None to fetch data until now.
            granularity (str): Granularity of the data. Default is 'm' (minute).

        Returns
        -------
            Dataframe with the fetched data in the format: timestamp (index), value (column).

        Raises
        ------
            WavefrontFetcherError: If there is an error fetching data from Wavefront
        """
        start = int(start.timestamp())
        if end:
            end = int(end.timestamp())
        if filters:
            _filters = " and ".join([f'{key}="{value}"' for key, value in filters.items()])
            query = f"ts({metric}, {_filters})"
        else:
            query = f"ts({metric}"
        LOGGER.info("Fetching data from Wavefront for query: %s", query)
        res = self._call_api(query, start, end, granularity)
        return self._format_results(res.to_dict())

    def raw_fetch(
        self,
        query: str,
        start: datetime,
        filters: Optional[dict] = None,
        end: Optional[datetime] = None,
        granularity: str = "m",
    ) -> pd.DataFrame:
        """
        Fetches data from Wavefront using a raw query, allowing for more complex queries.

        Args:
            query (str): Raw query to fetch data.
            start (datetime): Start time.
            filters (dict): Filters to apply to the query.
            end (datetime): End time. Set to None to fetch data until now.
            granularity (str): Granularity of the data. Default is 'm' (minute).

        Returns
        -------
            Dataframe with the fetched data in the format: timestamp (index), value (column).

        Raises
        ------
            WavefrontFetcherError:
                - If there is an error fetching data from Wavefront
                - If there is a key error in the query.

        >>> from datetime import datetime, timedelta
        ...
        >>> fetcher = WavefrontFetcher(url="https://miata.wavefront.com", api_token="6spd-manual")
        >>> df = fetcher.raw_fetch(
        ...     query="rawsum(ts(engine.rpm, gear='{gear}' and track='{track}'))",
        ...     start=datetime.now() - timedelta(minutes=5),
        ...     filters={"gear": "1", "track": "laguna_seca"},
        ...     end=datetime.now(),
        ... )
        """
        start = start.timestamp()
        if end:
            end = end.timestamp()

        try:
            query = query.format(**filters)
        except KeyError as key_err:
            raise WavefrontFetcherError(f"Key error in query: {key_err}") from key_err

        LOGGER.info("Fetching data from Wavefront for query: %s", query)
        qres = self._call_api(query, start, granularity, end)
        return self._format_results(qres.to_dict())
