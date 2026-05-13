import pandas as pd


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    """
    생활비용 모델은 아직 미완성.
    추후 모델 또는 CSV가 들어오면 이 함수만 교체하면 됨.
    """
    return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])