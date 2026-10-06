from pathlib import Path

import pandas as pd

from database import get_engine

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "social_media_ml.csv"
TABLE_NAME = "social_media_impact"


def main() -> None:
    engine = get_engine()
    df = pd.read_csv(CSV_PATH)
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    print(f"Uploaded {len(df)} rows to table '{TABLE_NAME}'")


if __name__ == "__main__":
    main()
