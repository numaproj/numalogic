from typing import Annotated

import typer

from numalogic.backtest._clifunc import univar_backtest, multivar_backtest
from numalogic.backtest._constants import DEFAULT_PROM_LOCALHOST

app = typer.Typer()


@app.command()
def univariate(
    namespace: Annotated[str, typer.Argument(help="Namespace name")],
    appname: Annotated[str, typer.Argument(help="Application name")],
    metric: Annotated[str, typer.Argument(help="The timeseries metric to analyze")],
    url: Annotated[
        str, typer.Option(envvar="PROM_URL", help="Endpoint URL for datafetching")
    ] = DEFAULT_PROM_LOCALHOST,
    lookback_days: Annotated[int, typer.Option(help="Number of days of data to fetch")] = 8,
):
    """CLI entry point for backtest run for a single metric."""
    univar_backtest(
        namespace=namespace, appname=appname, metric=metric, url=url, lookback_days=lookback_days
    )


@app.command()
def multivariate():
    """CLI entry point for backtest run for multiple metrics in a multivariate fashion."""
    multivar_backtest()
