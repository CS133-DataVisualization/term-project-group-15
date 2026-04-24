from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
PREMIUM_THRESHOLD = 20.0
LEAKAGE_PRONE_COLUMNS = [
    "mat_initial_price",
    "mat_final_price",
    "price_category",
    "mat_discount_percent",
    "discount_amount",
    "is_free",
]


def build_feature_matrix(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Create a numeric-only feature matrix from the dataset."""
    feature_df = df.drop(columns=target_columns, errors="ignore").copy()
    feature_df = feature_df.select_dtypes(include=[np.number, "bool"]).copy()

    bool_cols = feature_df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        feature_df[col] = feature_df[col].astype(int)

    return feature_df


def build_targets(df: pd.DataFrame, threshold: float = PREMIUM_THRESHOLD) -> tuple[pd.Series, pd.Series]:
    """Return regression and classification targets."""
    price = pd.to_numeric(df["mat_final_price"], errors="coerce").fillna(0.0)
    y_reg = price.rename("price_target")
    y_cls = (price >= threshold).astype(int).rename("is_premium")
    return y_reg, y_cls


def create_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Create train/test split shared across regression and classification tasks."""
    X = build_feature_matrix(df, target_columns=LEAKAGE_PRONE_COLUMNS)
    y_reg, y_cls = build_targets(df, threshold=PREMIUM_THRESHOLD)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    # Fit imputation on training data only to avoid leakage.
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_reg_train": y_reg_train.reset_index(drop=True),
        "y_reg_test": y_reg_test.reset_index(drop=True),
        "y_cls_train": y_cls_train.reset_index(drop=True),
        "y_cls_test": y_cls_test.reset_index(drop=True),
    }

def save_split_artifacts(splits: dict[str, pd.DataFrame | pd.Series], output_dir: Path) -> None:
    """Save split artifacts as CSV files for teammates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in splits.items():
        output_path = output_dir / f"{name}.csv"
        if isinstance(obj, pd.Series):
            obj.to_frame().to_csv(output_path, index=False)
        else:
            obj.to_csv(output_path, index=False)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    candidate_paths = [
        project_dir / "cleaned_games_dataset.csv",
        project_dir.parent / "cleaned_games_dataset.csv",
    ]
    input_path = next((path for path in candidate_paths if path.exists()), None)
    output_dir = project_dir / "ml_splits"

    if input_path is None:
        raise FileNotFoundError(
            "Could not find cleaned_games_dataset.csv in either the models folder "
            "or its parent Project Assignments folder."
        )

    df = pd.read_csv(input_path)
    splits = create_splits(df)
    save_split_artifacts(splits, output_dir)

    print("Saved split files to:", output_dir)
    print("Train rows:", len(splits["X_train"]), "| Test rows:", len(splits["X_test"]))
    print("Features available:", splits["X_train"].shape[1])
    print("Premium threshold:", PREMIUM_THRESHOLD)


if __name__ == "__main__":
    main()
