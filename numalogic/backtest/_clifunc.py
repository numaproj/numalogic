import shutil
from pathlib import Path
from typing import Union

import pandas as pd

from numalogic.backtest._prom import PromUnivarBacktester


def univar_backtest(namespace: str, appname: str, metric: str, url: str, lookback_days: int):
    """Run backtest for a single metric."""
    backtester = PromUnivarBacktester(url, namespace, appname, metric, lookback_days=lookback_days)
    df = backtester.read_data()
    backtester.train_models(df)
    out_df = backtester.generate_scores(df)
    backtester.save_plots(out_df)


def multivar_backtest(*_, **__):
    """Run backtest for multiple metrics in a multivariate fashion."""
    raise NotImplementedError


def clear_outputs(appname: str, metric: str, output_dir: str) -> None:
    """Clear the backtest output files."""
    _dir = PromUnivarBacktester.get_outdir(appname, metric, outdir=output_dir)
    print(f"Clearing backtest output files in {_dir}")
    shutil.rmtree(_dir, ignore_errors=False, onerror=None)


def train_models(
    data_file: Union[Path, str], col_name: str, train_ratio: float, output_dir: Union[Path, str]
):
    """Train models for the given data."""
    backtester = PromUnivarBacktester(
        "", "", "", col_name, test_ratio=(1 - train_ratio), output_dir=output_dir
    )

    df = pd.read_csv(data_file)
    df.set_index(["timestamp"], inplace=True)
    df.index = pd.to_datetime(df.index)

    backtester.train_models(df)


def generate_scores(
    data_file: Union[Path, str], col_name: str, model_path: Union[Path, str], test_ratio: float
):
    """Generate scores for the given data."""
    backtester = PromUnivarBacktester("", "", "", col_name, test_ratio=test_ratio)

    df = pd.read_csv(data_file)
    df.set_index(["timestamp"], inplace=True)
    df.index = pd.to_datetime(df.index)

    backtester.generate_scores(df, model_path=model_path)
