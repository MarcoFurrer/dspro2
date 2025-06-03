import os
import json
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from numerapi import NumerAPI as napi
from numerai_tools.scoring import numerai_corr, correlation_contribution
from typing import Literal
import json


class ModelRunner:
    def __init__(
        self,
        path_train: str = "data/train.parquet",
        path_val: str = "data/validation.parquet",
        path_meta_model: str = "data/meta_model.parquet",
        path_features: str = "data/features.json",
        output_path: str = "exports",
        batch_size: int = 32,
        subset_features: Literal["small", "medium", "all"] = "small",
        model=None,
    ):
        self.path_train = path_train
        self.train_dataset = pd.read_parquet(self.path_train)
        self.path_val = path_val
        self.validation_dataset = pd.read_parquet(self.path_val)
        self.path_meta_model = path_meta_model
        self.meta_model = pd.read_parquet(self.path_meta_model)
        self.path_features = path_features
        self.feature_set = json.load(open(self.path_features, "r", encoding="utf-8"))
        self.output_path = output_path
        self.batch_size = batch_size
        self.subset_features = subset_features
        self.feature_count = None
        self.model = model
        self.target_set = None
        self._validation = None

        os.makedirs(output_path, exist_ok=True)

    @classmethod
    def download_data(cls, data_version="v5.0"):
        # list the datasets and available versions
        all_datasets = napi.list_datasets()
        dataset_versions = list(set(d.split("/")[0] for d in all_datasets))
        print("Available versions:\n", dataset_versions)

        # Print all files available for download for our version
        current_version_files = [f for f in all_datasets if f.startswith(data_version)]
        print("Available", data_version, "files:\n", current_version_files)

        # download the feature metadata file
        napi.download_dataset(f"{data_version}/features.json")
        napi.download_dataset(f"{data_version}/validation.parquet")

    def plot_data(self):
        # Plot the number of rows per era
        self.train_dataset.groupby("era").size().plot(
            title="Number of rows per era", figsize=(5, 3), xlabel="Era"
        )

        # Plot density histogram of the target
        self.train_dataset["target"].plot(
            kind="hist",
            title="Target",
            figsize=(5, 3),
            xlabel="Value",
            density=True,
            bins=50,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        first_era = self.train_dataset[
            self.train_dataset["era"] == self.train_dataset["era"].unique()[0]
        ]
        last_era = self.train_dataset[
            self.train_dataset["era"] == self.train_dataset["era"].unique()[-1]
        ]
        last_era[self.feature_set[-1]].plot(
            title="5 equal bins", kind="hist", density=True, bins=50, ax=ax1
        )
        first_era[self.feature_set[-1]].plot(
            title="missing data", kind="hist", density=True, bins=50, ax=ax2
        )

    def export_model(self):
        """Simple model export"""
        model_path = os.path.join(self.output_path, "model.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

    def train(self, validation_split=0.1, epochs=5):
        """Train the model using memory-efficient batch processing"""

        X = self.train_dataset[
            self.feature_set["feature_sets"][self.subset_features]
        ].values
        y = self.train_dataset["target"].values.astype(np.float16)

        # Calculate split indices
        total_samples = len(X)
        val_samples = int(total_samples * validation_split)
        train_samples = total_samples - val_samples

        # Split data
        X_train = X[:train_samples]
        y_train = y[:train_samples]
        X_val = X[train_samples:]
        y_val = y[train_samples:]

        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")

        self.model.summary()

        # Train the model
        print(
            f"Training model for {epochs} epochs with batch size {self.batch_size}..."
        )
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=0.0001, verbose=1
                ),
            ],
            verbose=1,
        )

        # Clean up memory
        del X, y, X_train, y_train, X_val, y_val

        # Export model
        self.export_model()

        return self.model, history

    def predict(self, X):
        """Generate continuous predictions for regression"""
        # Ensure proper data type for continuous features
        X = np.ascontiguousarray(X, dtype=np.float16)

        # Get raw predictions
        raw_preds = self.model.predict(X)

        return raw_preds.squeeze()

    def validate_model(self):
        """Validate the model on validation dataset"""

        print("Preparing validation data...")

        # Use existing validation_dataset - filter for validation data type
        validation = self.validation_dataset[
            self.validation_dataset["data_type"] == "validation"
        ]

        # Downsample to every 4th era
        unique_eras = validation["era"].unique()
        validation = validation[validation["era"].isin(unique_eras[::4])]
        print(f"Using {len(unique_eras[::4])} eras for validation")

        # Get last training era and embargo
        last_train_era = int(self.train_dataset["era"].max())
        validation = validation[
            ~validation["era"].isin(
                [
                    str(era).zfill(4)
                    for era in range(last_train_era + 1, last_train_era + 5)
                ]
            )
        ]

        print(f"Final validation samples: {len(validation):,}")
        X_val = validation[
            self.feature_set["feature_sets"][self.subset_features]
        ].values.astype(np.float16)
        validation["prediction"] = self.model.predict(X_val, verbose=1).squeeze()

        return validation

    def performance_eval(self):
        if self._validation is None:
            print("Please run validation before evaluating the performance!")
            return None

        # Compute the per-era corr between our predictions and the target values
        per_era_corr = (
            self.meta_model["meta_model"]
            .groupby("era")
            .apply(
                lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
            )
        )

        # Compute the per-era mmc between our predictions, the meta model, and the target values
        per_era_mmc = (
            self.meta_model["meta_model"]
            .dropna()
            .groupby("era")
            .apply(
                lambda x: correlation_contribution(
                    x[["prediction"]], x["meta_model"], x["target"]
                )
            )
        )

        # Plot the per-era correlation
        per_era_corr.plot(
            title="Validation CORR",
            kind="bar",
            figsize=(8, 4),
            xticks=[],
            legend=False,
            snap=False,
        )
        per_era_mmc.plot(
            title="Validation MMC",
            kind="bar",
            figsize=(8, 4),
            xticks=[],
            legend=False,
            snap=False,
        )

        # Plot the cumulative per-era correlation
        per_era_corr.cumsum().plot(
            title="Cumulative Validation CORR",
            kind="line",
            figsize=(8, 4),
            legend=False,
        )
        per_era_mmc.cumsum().plot(
            title="Cumulative Validation MMC", kind="line", figsize=(8, 4), legend=False
        )

        # Compute performance metrics
        corr_mean = per_era_corr.mean()
        corr_std = per_era_corr.std(ddof=0)
        mmc_mean = per_era_mmc.mean()
        mmc_std = per_era_mmc.std(ddof=0)

        pd.DataFrame(
            {
                "mean": [corr_mean, mmc_mean],
                "std": [corr_std, mmc_std],
                "sharpe": [corr_mean / corr_std, mmc_mean / mmc_std],
                "max_drawdown": [
                    (
                        per_era_corr.cumsum().expanding(min_periods=1).max()
                        - per_era_corr.cumsum()
                    ).max(),
                    (
                        per_era_mmc.cumsum().expanding(min_periods=1).max()
                        - per_era_mmc.cumsum()
                    ).max(),
                ],
            },
            index=["CORR", "MMC"],
        ).T
