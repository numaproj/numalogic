import logging
import shutil
from pathlib import Path
from typing import Annotated
from typing import Optional

import typer

import numalogic.backtest._bt as bt
from numalogic.backtest._constants import DEFAULT_OUTPUT_DIR
from numalogic.backtest._clifunc import clear_outputs, train_models, generate_scores

logging.basicConfig(level=logging.INFO)


app = typer.Typer()
app.add_typer(bt.app, name="backtest")


@app.command()
def clear(
    appname: Optional[str] = None,
    metric: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    all_: Annotated[bool, typer.Option("--all")] = False,
):
    """CLI entrypoint for clearing the backtest output files."""
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
    train_ratio: Annotated[float, typer.Option()] = 0.9,
    output_dir: Annotated[Optional[Path], typer.Option()] = DEFAULT_OUTPUT_DIR,
):
    """CLI entrypoint for training models for the given data."""
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    train_models(
        data_file=data_file, col_name=col_name, train_ratio=train_ratio, output_dir=output_dir
    )


@app.command()
def score(
    data_file: Annotated[Optional[Path], typer.Option()] = None,
    col_name: Annotated[Optional[str], typer.Option()] = None,
    model_path: Annotated[Optional[Path], typer.Option()] = None,
    test_ratio: Annotated[float, typer.Option()] = 1.0,
):
    """CLI entrypoint for generating scores for the given data."""
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    generate_scores(
        data_file=data_file, col_name=col_name, model_path=model_path, test_ratio=test_ratio
    )
