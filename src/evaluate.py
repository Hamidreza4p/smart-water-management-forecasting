"""Leakage-safe evaluation for Bilancino Lake forecasting.

The script predicts the next observation from lagged measurements only. It uses a
chronological holdout and time-series cross-validation; no future target values are
used as features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


START_DATE = "2002-06-03"
REQUIRED_COLUMNS = [
    "Date",
    "Mean_Rainfall",
    "Mean_Temp",
    "Actual_Flow_Rate",
    "Actual_Lake_Level",
]
TARGETS = ["Actual_Lake_Level", "Actual_Flow_Rate"]
LAGS = (1, 3, 7, 14, 28)
WINDOWS = (7, 28)


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("The prepared dataset contains missing values.")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(START_DATE) + pd.to_timedelta(
        frame["Date"], unit="D"
    )
    frame = frame.set_index("Date").sort_index().asfreq("D")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Date index must be unique and increasing.")
    return frame


def build_supervised_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)

    # Every measurement is shifted before use, so the row at time t only contains
    # information observed at t-1 or earlier.
    for column in frame.columns:
        for lag in LAGS:
            features[f"{column}_lag_{lag}"] = frame[column].shift(lag)
        shifted = frame[column].shift(1)
        for window in WINDOWS:
            features[f"{column}_mean_{window}"] = shifted.rolling(window).mean()
            features[f"{column}_std_{window}"] = shifted.rolling(window).std()

    day_of_year = frame.index.day_of_year
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["target"] = frame[target]
    return features.dropna()


def metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def evaluate_target(data: pd.DataFrame, target: str, output_dir: Path) -> dict:
    supervised = build_supervised_frame(data, target)
    split = int(len(supervised) * 0.8)
    train = supervised.iloc[:split]
    test = supervised.iloc[split:]
    x_train = train.drop(columns="target")
    y_train = train["target"]
    x_test = test.drop(columns="target")
    y_test = test["target"]

    persistence = x_test[f"{target}_lag_1"].to_numpy()

    search = GridSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        param_grid={
            "n_estimators": [150, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.08],
            "subsample": [0.8],
        },
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=4),
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    xgb_prediction = search.best_estimator_.predict(x_test)

    arima = ARIMA(y_train, order=(5, 1, 0)).fit()
    arima_prediction = np.asarray(arima.forecast(steps=len(y_test)))

    exog_columns = ["Mean_Rainfall_lag_1", "Mean_Temp_lag_1"]
    if target == "Actual_Flow_Rate":
        exog_columns.append("Actual_Lake_Level_lag_1")
    else:
        exog_columns.append("Actual_Flow_Rate_lag_1")
    sarimax = SARIMAX(
        y_train,
        exog=x_train[exog_columns],
        order=(5, 1, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarimax_prediction = np.asarray(
        sarimax.forecast(steps=len(y_test), exog=x_test[exog_columns])
    )

    results = {
        "target": target,
        "observations": int(len(supervised)),
        "train_observations": int(len(train)),
        "test_observations": int(len(test)),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "persistence": metrics(y_test, persistence),
        "xgboost": {
            **metrics(y_test, xgb_prediction),
            "best_parameters": search.best_params_,
            "best_cv_rmse": float(-search.best_score_),
        },
        "arima_5_1_0": metrics(y_test, arima_prediction),
        "sarimax_5_1_0": metrics(y_test, sarimax_prediction),
    }

    plot_count = min(365, len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index[-plot_count:], y_test.iloc[-plot_count:], label="Actual")
    plt.plot(
        y_test.index[-plot_count:],
        xgb_prediction[-plot_count:],
        label="XGBoost",
        alpha=0.85,
    )
    plt.plot(
        y_test.index[-plot_count:],
        persistence[-plot_count:],
        label="Persistence baseline",
        alpha=0.7,
    )
    plt.title(f"{target}: final {plot_count} test observations")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{target.lower()}_forecast.png", dpi=160)
    plt.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/lakes_fe.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    results = {
        "dataset": {
            "rows": int(len(data)),
            "start": str(data.index.min().date()),
            "end": str(data.index.max().date()),
            "split": "chronological 80/20",
            "feature_policy": "lagged observations only",
        },
        "evaluations": [evaluate_target(data, target, args.output) for target in TARGETS],
    }
    result_path = args.output / "metrics.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

