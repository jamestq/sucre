from pathlib import Path
from typing import Annotated
import typer, traceback, yaml, pandas as pd

from sucre import run as runner

from .utils import change_dir

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
        data_folder = config_data.pop("data_folder", None) 
        df_list: list[pd.DataFrame] = []        
        with change_dir(config.parent / data_folder if data_folder else Path.cwd()):            
            for command, data in config_data.items():
                df_list = runner(command, data, df_list)
    except Exception as e:
        typer.echo(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    app()
