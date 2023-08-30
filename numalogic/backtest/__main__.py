import typer
from rich import print


app = typer.Typer()


@app.command()
def backtest(url: str, namespace: str, metrics: list[str], dataconnector: str = "prometheus"):
    pass


@app.command()
def train():
    print("Training")


@app.command()
def infer():
    print("Infering")


if __name__ == "__main__":
    typer.run(app)
