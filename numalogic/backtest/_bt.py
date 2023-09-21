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

import os
from typing import Annotated, Optional

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
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
):
    """CLI entry point for backtest run for a single metric."""
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), ".btoutput")
    univar_backtest(
        namespace=namespace,
        appname=appname,
        metric=metric,
        url=url,
        lookback_days=lookback_days,
        output_dir=output_dir,
    )


@app.command()
def multivariate():
    """CLI entry point for backtest run for multiple metrics in a multivariate fashion."""
    multivar_backtest()
