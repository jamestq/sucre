import typer
from pathlib import Path
from pandasgui import show
from typing import Annotated

from sucre import preprocessor

app = typer.Typer()

@app.command()
def hello():
    typer.echo("Hello, Sucre!")

@app.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
def preprocess(
    command: Annotated[str, typer.Option("--command", "-c", help="The preprocessing command to execute.")],
    input: Annotated[str, typer.Option("--input", "-i", help="The input file path.")],
    ctx: typer.Context,
):
    try:
        path: Path = Path(input)
        df = preprocessor.run(command, path)
        show(df)
    except Exception as e:
        typer.echo(f"Error: {e}")


if __name__ == "__main__":
    app()
