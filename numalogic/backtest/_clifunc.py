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
import shutil
from pathlib import Path
from typing import Union

import pandas as pd

from numalogic.backtest._prom import PromUnivarBacktester
from numalogic.tools.exceptions import DataFormatError


logger = logging.getLogger(__name__)


def univar_backtest(
    namespace: str, appname: str, metric: str, url: str, lookback_days: int, output_dir: str
):
    """Run backtest for a single metric."""
    backtester = PromUnivarBacktester(
        url, namespace, appname, metric, lookback_days=lookback_days, output_dir=output_dir
    )
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
    logger.info("Clearing backtest output files in %s", _dir)
    shutil.rmtree(_dir, ignore_errors=False, onerror=None)


def train_models(
    data_file: Union[Path, str],
    col_name: str,
    ts_col_name: str,
    train_ratio: float,
    output_dir: Union[Path, str],
):
    """Train models for the given data."""
    backtester = PromUnivarBacktester(
        "", "", "", col_name, test_ratio=(1 - train_ratio), output_dir=output_dir
    )

    df = pd.read_csv(data_file)
    try:
        df.set_index([ts_col_name], inplace=True)
    except KeyError:
        raise DataFormatError(f"Timestamp column {ts_col_name} not found in the data!") from None

    df.index = pd.to_datetime(df.index)
    backtester.train_models(df)


def generate_scores(
    data_file: Union[Path, str],
    col_name: str,
    ts_col_name: str,
    model_path: Union[Path, str],
    test_ratio: float,
):
    """Generate scores for the given data."""
    backtester = PromUnivarBacktester("", "", "", col_name, test_ratio=test_ratio)

    df = pd.read_csv(data_file)
    try:
        df.set_index([ts_col_name], inplace=True)
    except KeyError:
        raise DataFormatError(f"Timestamp column {ts_col_name} not found in the data!") from None

    df.index = pd.to_datetime(df.index)
    backtester.generate_scores(df, model_path=model_path)
