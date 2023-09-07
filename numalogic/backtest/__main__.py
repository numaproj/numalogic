import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from typing import Annotated

import numalogic.backtest._bt as dim
from numalogic.backtest._constants import DEFAULT_OUTPUT_DIR
from numalogic.backtest._prom import PromUnivarBacktester
from numalogic.udfs import set_logger

set_logger()


app = typer.Typer()
app.add_typer(dim.app, name="backtest")


@app.command()
def clear(
    appname: Optional[str] = None,
    metric: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    all_: Annotated[bool, typer.Option("--all")] = False,
):
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

    _dir = PromUnivarBacktester.get_outdir(appname, metric, outdir=output_dir)
    print(f"Clearing backtest output files in {_dir}")
    shutil.rmtree(_dir, ignore_errors=False, onerror=None)


@app.command()
def train(
    data_file: Annotated[Optional[Path], typer.Option()] = None,
    col_name: Annotated[Optional[str], typer.Option()] = None,
    train_ratio: Annotated[float, typer.Option()] = 0.9,
    output_dir: Annotated[Optional[Path], typer.Option()] = DEFAULT_OUTPUT_DIR,
):
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    backtester = PromUnivarBacktester(
        "", "", "", col_name, test_ratio=(1 - train_ratio), output_dir=output_dir
    )

    df = pd.read_csv(data_file)
    df.set_index(["timestamp"], inplace=True)
    df.index = pd.to_datetime(df.index)

    backtester.train_models(df)


@app.command()
def score(
    data_file: Annotated[Optional[Path], typer.Option()] = None,
    col_name: Annotated[Optional[str], typer.Option()] = None,
    model_path: Annotated[Optional[Path], typer.Option()] = None,
    test_ratio: Annotated[float, typer.Option()] = 1.0,
):
    if (data_file is None) or (col_name is None):
        print("No data file or column name provided!")
        raise typer.Abort()

    backtester = PromUnivarBacktester("", "", "", col_name, test_ratio=test_ratio)

    df = pd.read_csv(data_file)
    df.set_index(["timestamp"], inplace=True)
    df.index = pd.to_datetime(df.index)

    backtester.generate_scores(df, model_path=model_path)


if __name__ == "__main__":
    app()
