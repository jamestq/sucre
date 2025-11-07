import pandas as pd

from sucre.preprocessor.base import read

__all__ = ["predict"]

def predict(
    df: pd.DataFrame | None = None, **kwargs
):
    df = read(df, **kwargs)
    