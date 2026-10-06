import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "database"))
from database import get_engine

NUMERIC_FEATURES = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
]
TARGET = "Academic_Performance_GPA"
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_gpa.joblib"


def load_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql_table("social_media_impact", engine)
    df.columns = df.columns.map(str)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main() -> None:
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2:   {r2_score(y_test, y_pred):.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
