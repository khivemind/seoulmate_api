from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EXPENSES_PATH = BASE_DIR / "app/data/생활비용지수.csv"


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    """
    CSV에서 읽어서 데이터 가공 후 반환 
    """

    try:
        df = pd.read_csv(EXPENSES_PATH, encoding="utf-8-sig")

        df = df[(df["년도"] == year) & (df["월"] == month)]

        if df.empty:
            print(f"[expense] 데이터 없음: year={year}, month={month}")
            return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

        df = df.rename(columns={
            "행정동코드":"code",
            "행정동_명칭":"dong",
            "자치구_명칭":"gu",
            "생활비용지수_등급":"grade",
            "생활비용지수":"score"
            })

        return df[["code", "dong", "gu", "grade", "score"]].copy()

    except FileNotFoundError:
        print("[expense] 파일을 찾을 수 없습니다.")
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    except UnicodeDecodeError:
        print("[expense] 인코딩 오류 발생")
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])
         
    except pd.errors.EmptyDataError:
        print("[expense] 파일이 비어 있습니다.")
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    except pd.errors.ParserError as e:
        print("[expense] CSV 파싱 오류:", e)
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    except Exception as e:
        print("[expense] 알 수 없는 오류:", e)               
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

