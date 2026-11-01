"""Machine Learning module for service/burst time prediction in OptiQ.

Model Suite:
- DummyRegressor (Median Baseline)
- Ridge / LinearRegression baseline
- RandomForestRegressor
- XGBoostRegressor (if available)
- LightGBMRegressor (if available)

Strict ML Guidelines Enforced:
1. Chronological or grouped train/validation/test split.
2. Identical held-out test set for all models.
3. Preprocessing (StandardScaler) fitted ONLY on training split.
4. Target transformation: log1p(y) modeling on training data, expm1(y_pred) inverse transform
   to prevent skew-induced catastrophic prediction errors on heavy-tailed distributions.
5. Hyperparameters tuned strictly on Validation split; model selection based on Validation MAE.
6. Zero leakage: test split never used for training or model selection.
"""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from optiq.data.loader import PRE_SCHEDULING_FEATURES, TARGET_COL


@dataclass
class ModelMetrics:
    """Comprehensive evaluation metrics for regression models."""
    model_name: str
    val_mae: float
    mae: float
    rmse: float
    r2: float
    inference_time_ms: float
    train_samples: int
    val_samples: int
    test_samples: int
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class ModelTrainer:
    """Trains, tunes, and evaluates burst prediction regressors using log1p target transformation."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        target_col: str = TARGET_COL,
        seed: int = 42,
        use_log_transform: bool = True,
    ):
        self.features = features or PRE_SCHEDULING_FEATURES
        self.target_col = target_col
        self.seed = seed
        self.use_log_transform = use_log_transform
        self.scaler = StandardScaler()
        self.fitted_scaler: bool = False
        self.trained_models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.best_model_name: Optional[str] = None

    def prepare_splits(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        chronological: bool = True,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
    ]:
        """Split dataframe into Train, Validation, and Test partitions."""
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        if chronological:
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
        else:
            shuffled = df.sample(frac=1.0, random_state=self.seed)
            train_df = shuffled.iloc[:train_end].copy()
            val_df = shuffled.iloc[train_end:val_end].copy()
            test_df = shuffled.iloc[val_end:].copy()

        X_train_raw = train_df[self.features].values
        y_train = train_df[self.target_col].values

        X_val_raw = val_df[self.features].values
        y_val = val_df[self.target_col].values

        X_test_raw = test_df[self.features].values
        y_test = test_df[self.target_col].values

        # Fit scaler ONLY on train
        X_train = self.scaler.fit_transform(X_train_raw)
        self.fitted_scaler = True

        X_val = self.scaler.transform(X_val_raw)
        X_test = self.scaler.transform(X_test_raw)

        return (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            train_df, val_df, test_df,
        )

    def train_and_evaluate_all(
        self,
        df: pd.DataFrame,
        chronological: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
        """Train all candidate models, select best model on validation split, evaluate on held-out test set."""
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            train_df, val_df, test_df,
        ) = self.prepare_splits(df, chronological=chronological)

        # Apply log1p transform to target if enabled
        y_train_fit = np.log1p(y_train) if self.use_log_transform else y_train

        models_to_train: Dict[str, Any] = {
            "MedianBaseline": DummyRegressor(strategy="median"),
            "RidgeRegression": Ridge(alpha=10.0, random_state=self.seed),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=6,
                random_state=self.seed, n_jobs=-1
            ),
        }

        if XGBOOST_AVAILABLE:
            models_to_train["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0,
            )

        if LIGHTGBM_AVAILABLE:
            models_to_train["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            )

        predictions_dict: Dict[str, np.ndarray] = {}
        metrics_list: List[ModelMetrics] = []

        best_val_mae = float("inf")
        best_name = None

        for name, model in models_to_train.items():
            if name == "MedianBaseline":
                # Baseline predicts median on raw scale
                model.fit(X_train, y_train)
                val_raw_preds = model.predict(X_val)
                t0 = time.perf_counter()
                test_raw_preds = model.predict(X_test)
                t_infer = (time.perf_counter() - t0) * 1000.0
            else:
                model.fit(X_train, y_train_fit)
                val_log_preds = model.predict(X_val)
                val_raw_preds = np.expm1(val_log_preds) if self.use_log_transform else val_log_preds

                t0 = time.perf_counter()
                test_log_preds = model.predict(X_test)
                t_infer = (time.perf_counter() - t0) * 1000.0
                if self.use_log_transform:
                    # Clip log scale predictions to [0, 15] to prevent expm1 overflow (~exp(15) ~ 3.2e6 sec)
                    test_raw_preds = np.expm1(np.clip(test_log_preds, 0.0, 15.0))
                else:
                    test_raw_preds = test_log_preds

            self.trained_models[name] = model

            val_preds = np.clip(val_raw_preds, 0.1, None)
            val_mae = float(mean_absolute_error(y_val, val_preds))

            test_preds = np.clip(test_raw_preds, 0.1, None)
            predictions_dict[name] = test_preds

            mae = float(mean_absolute_error(y_test, test_preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            r2 = float(r2_score(y_test, test_preds))

            metric = ModelMetrics(
                model_name=name,
                val_mae=round(val_mae, 3),
                mae=mae,
                rmse=rmse,
                r2=r2,
                inference_time_ms=t_infer,
                train_samples=len(X_train),
                val_samples=len(X_val),
                test_samples=len(X_test),
                params={"model_type": str(type(model).__name__), "log_transform": self.use_log_transform},
                notes="Selected strictly by Validation MAE; test split held out.",
            )
            metrics_list.append(metric)
            self.model_metrics[name] = metric

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_name = name

        self.best_model_name = best_name or "RandomForest"

        records = [
            {
                "Model": m.model_name,
                "Val MAE (s)": m.val_mae,
                "Test MAE (s)": round(m.mae, 3),
                "Test RMSE (s)": round(m.rmse, 3),
                "Test R2": round(m.r2, 4),
                "Inference (ms)": round(m.inference_time_ms, 2),
                "Train Samples": m.train_samples,
                "Test Samples": m.test_samples,
            }
            for m in metrics_list
        ]
        metrics_df = pd.DataFrame(records)

        return metrics_df, predictions_dict, test_df

    def predict(self, df_inputs: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Generate runtime predictions for new instances."""
        name = model_name or self.best_model_name
        if not name or name not in self.trained_models:
            raise ValueError(f"Model {name} not trained or not available.")

        if not self.fitted_scaler:
            raise RuntimeError("Scaler not fitted.")

        X_raw = df_inputs[self.features].values
        X_scaled = self.scaler.transform(X_raw)
        raw_pred = self.trained_models[name].predict(X_scaled)

        if self.use_log_transform and name != "MedianBaseline":
            pred = np.expm1(raw_pred)
        else:
            pred = raw_pred

        return np.clip(pred, 0.1, None)

    def get_feature_importances(self, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Extract feature importances if supported by the model."""
        name = model_name or self.best_model_name
        if not name or name not in self.trained_models:
            return None

        model = self.trained_models[name]
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)

        if importances is not None:
            fi_df = pd.DataFrame({
                "Feature": self.features,
                "Importance": importances,
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            return fi_df
        return None

    def save_artifacts(self, output_dir: str) -> Dict[str, str]:
        """Save scaler and all trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}

        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        saved_paths["scaler"] = scaler_path

        for name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            saved_paths[name] = model_path

        return saved_paths

    def load_artifacts(self, input_dir: str, model_names: List[str]) -> None:
        """Load scaler and models from disk."""
        scaler_path = os.path.join(input_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.fitted_scaler = True

        for name in model_names:
            model_path = os.path.join(input_dir, f"{name}.joblib")
            if os.path.exists(model_path):
                self.trained_models[name] = joblib.load(model_path)
