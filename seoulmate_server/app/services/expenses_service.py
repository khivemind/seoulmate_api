import pandas as pd


def get_heatmap(year: int, month: int):
    return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])