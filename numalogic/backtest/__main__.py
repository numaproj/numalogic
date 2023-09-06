import shutil
from typing import Optional

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
def train():
    pass


@app.command()
def score():
    pass


if __name__ == "__main__":
    app()
