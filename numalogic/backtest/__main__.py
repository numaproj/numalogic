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
import os
import shutil
from pathlib import Path
from typing import Annotated
from typing import Optional

import typer

import numalogic.backtest._bt as bt
from numalogic.backtest._clifunc import clear_outputs, train_models, generate_scores

logging.basicConfig(level=logging.INFO)


app = typer.Typer()
app.add_typer(bt.app, name="backtest")


@app.command()
def clear(
    appname: Optional[str] = None,
    metric: Optional[str] = None,
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
):
    """CLI entrypoint for clearing the backtest output files."""
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), ".btoutput")

    if all_:
        print(f"Clearing all the backtest output files in {output_dir}")
        try:
            shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
        except FileNotFoundError:
            pass
        return

    if not (appname and metric):
        _msg = "Both appname and metric needs to be provided!"
        print(_msg)
        return

    clear_outputs(appname=appname, metric=metric, output_dir=output_dir)


@app.command()
def train(
    data_file: Annotated[Optional[Path], typer.Option()] = None,
    col_name: Annotated[Optional[str], typer.Option()] = None,
    ts_col_name: Annotated[str, typer.Option()] = "timestamp",
    train_ratio: Annotated[float, typer.Option()] = 0.9,
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
):
    """CLI entrypoint for training models for the given data."""
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    if not output_dir:
        output_dir = os.path.join(os.getcwd(), ".btoutput")

    train_models(
        data_file=data_file,
        col_name=col_name,
        ts_col_name=ts_col_name,
        train_ratio=train_ratio,
        output_dir=output_dir,
    )


@app.command()
def score(
    data_file: Annotated[Optional[Path], typer.Option()] = None,
    col_name: Annotated[Optional[str], typer.Option()] = None,
    ts_col_name: Annotated[str, typer.Option()] = "timestamp",
    model_path: Annotated[Optional[Path], typer.Option()] = None,
    test_ratio: Annotated[float, typer.Option()] = 1.0,
):
    """CLI entrypoint for generating scores for the given data."""
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    generate_scores(
        data_file=data_file,
        col_name=col_name,
        ts_col_name=ts_col_name,
        model_path=model_path,
        test_ratio=test_ratio,
    )
