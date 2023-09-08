from typing import Annotated

import typer

from numalogic.backtest._constants import DEFAULT_PROM_LOCALHOST
from numalogic.backtest._prom import PromUnivarBacktester

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
    backtester = PromUnivarBacktester(url, namespace, appname, metric, lookback_days=lookback_days)
    df = backtester.read_data()
    backtester.train_models(df)
    out_df = backtester.generate_scores(df)
    backtester.save_plots(out_df)


@app.command()
def multivariate():
    raise NotImplementedError
