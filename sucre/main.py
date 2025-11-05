from pathlib import Path
from typing import Annotated
import typer, traceback, yaml

from sucre import run as runner

app = typer.Typer()

@app.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
def run(    
    config: Annotated[Path, typer.Argument(help="The path to the configuration file.")],
):
    try:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)
        for command, data in config_data.items():
            runner(command, data)
    except Exception as e:
        typer.echo(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    app()
