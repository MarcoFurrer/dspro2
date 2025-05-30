import os
import json
import matplotlib.pyplot as plt
import cloudpickle
import pyarrow.parquet as pq

import tensorflow as tf
import pandas as pd
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from numerapi import NumerAPI as napi
from numerai_tools.scoring import numerai_corr, correlation_contribution
from config import *
from typing import Literal


class ModelRunner:
    def __init__(
        self,
        path_train: str = "data/train.parquet",
        path_val: str = "data/validation.parquet",
        path_metadata: str = "data/features.json",
        path_meta_model: str = "data/meta_model.parquet",
        path_features: str = "data/features.json",
        data_version: str = "v5.0",
        output_path: str = "exports",
        batch_size: int = 64,
        subset_features: Literal["small", "medium", "all"] = "small",
        model=None,
    ):
        self.path_train = path_train
        self.path_val = path_val
        self.path_metadata = path_metadata
        self.path_meta_model = path_meta_model
        self.path_features = path_features
        self.data_version = data_version
        self.output_path = output_path
        self.batch_size = batch_size
        self.subset_features = subset_features
        self.feature_count = None
        self.n_categories = 5  # Categories 0-4
        self.model = model  # Store the external model if provided
        os.makedirs(output_path, exist_ok=True)
        self.feature_set = None
        self.target_set = None
        self.target_mapping = {
            0.0: 0,
            0.25: 1,
            0.5: 2,
            0.75: 3,
            1.0: 4,
        }  # Map float targets to integers
        self.inverse_target_mapping = {
            0: 0.0,
            1: 0.25,
            2: 0.5,
            3: 0.75,
            4: 1.0,
        }  # For converting back
        self._validation = None
        self.train_dataset = None
        self.validation_dataset = None

    def _download_data(self):
        # list the datasets and available versions
        all_datasets = napi.list_datasets()
        dataset_versions = list(set(d.split("/")[0] for d in all_datasets))
        print("Available versions:\n", dataset_versions)

        # Maybe add this as parameter later to change versions
        # Set data version to one of the latest datasets
        DATA_VERSION = "v5.0"

        # Print all files available for download for our version
        current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]
        print("Available", DATA_VERSION, "files:\n", current_version_files)

        # download the feature metadata file
        napi.download_dataset(f"{DATA_VERSION}/features.json")
        napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

        self.data_path_train = f"{DATA_VERSION}/features.json"
        self.data_path_val = f"{DATA_VERSION}/validation.parquet"

    def _get_dataset_info(self):
        """Get basic information about the dataset"""
        # read the metadata and display
        feature_metadata = json.load(open(self.data_path_metadata))
        for metadata in feature_metadata:
            print(metadata, len(feature_metadata[metadata]))

        feature_sets = feature_metadata["feature_sets"]
        for feature_set in ["small", "medium", "all"]:
            print(feature_set, len(feature_sets[feature_set]))

        feature_set = feature_sets[self._subset_features]
        self._feature_set = feature_set

        # Open the parquet file
        parquet_file = pq.ParquetFile(self.data_path_train)

        # Read only the first row group for a quick sample
        df_sample = pd.read_parquet(
            self.data_path_train, columns=["era", "target"] + self._feature_set
        )

        df_sample = df_sample.head(1)
        # Identify feature and target columns
        feature_cols = [col for col in df_sample.columns if col.startswith("feature_")]
        target_cols = [col for col in df_sample.columns if col.startswith("target")]

        total_rows = parquet_file.metadata.num_rows

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.feature_count = len(feature_cols)
        self.total_rows = total_rows

        print(f"Dataset has {total_rows:,} rows")
        print(
            f"Found {self.feature_count} feature columns and {len(target_cols)} target columns"
        )

    def plot_data(self):
        # Plot the number of rows per era
        self._train.groupby("era").size().plot(
            title="Number of rows per era", figsize=(5, 3), xlabel="Era"
        )

        # Plot density histogram of the target
        train["target"].plot(
            kind="hist",
            title="Target",
            figsize=(5, 3),
            xlabel="Value",
            density=True,
            bins=50,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        first_era = train[train["era"] == train["era"].unique()[0]]
        last_era = train[train["era"] == train["era"].unique()[-1]]
        last_era[feature_set[-1]].plot(
            title="5 equal bins", kind="hist", density=True, bins=50, ax=ax1
        )
        first_era[feature_set[-1]].plot(
            title="missing data", kind="hist", density=True, bins=50, ax=ax2
        )

    def export_model(self):
        """Simple model export"""
        model_path = os.path.join(self.output_path, "model.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

    def generator(self):
        parquet_file = pq.ParquetFile(data_path)

        # Use smaller read batches to reduce memory pressure
        read_batch_size = min(10000, self.total_rows // 50)

        for batch in parquet_file.iter_batches(batch_size=read_batch_size):
            df_batch = batch.to_pandas()

            # Extract features and target with explicit alignment
            X_batch = df_batch[self.feature_cols].values
            X_batch = np.ascontiguousarray(X_batch, dtype=np.float32)

            # Convert target values to uint8 integers (0-4) with explicit alignment
            # We keep the targert value in its original form
            #
            y_batch = df_batch[self.target_cols].values.squeeze()
            y_batch = np.ascontiguousarray(y_batch, dtype=np.float32)

            # Map float values to integers
            """
                for float_val, int_val in self.target_mapping.items():
                    y_batch[np.isclose(y_float, float_val)] = int_val
                """

            # Yield batches with explicit alignment
            for i in range(0, len(X_batch), self.batch_size):
                end_idx = min(i + self.batch_size, len(X_batch))
                # Create properly aligned copies
                x = np.ascontiguousarray(X_batch[i:end_idx])
                y = np.ascontiguousarray(y_batch[i:end_idx])
                yield x, y

            # Free memory
            del df_batch, X_batch, y_batch
            gc.collect()

    def create_dataset_pipeline(self, data_path, is_training=True):
        """Create an efficient TF dataset pipeline that processes data in batches"""
        output_signature = (
            tf.TensorSpec(shape=(None, self.feature_count), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )

        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            self.generator, output_signature=output_signature
        )

        # Configure dataset for performance - reduce shuffle buffer size
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if is_training:
            # Use a smaller shuffle buffer to avoid memory issues
            dataset = dataset.shuffle(buffer_size=1000)

        return dataset

    def train(self, validation_split=0.1, epochs=5):
        """Train the model using memory-efficient batch processing"""

        # Create full dataset pipeline
        print("Creating dataset pipeline...")
        self.train_dataset = self.create_dataset_pipeline(self.data_path_train)

        train_size = int(self.total_rows * (1 - validation_split))

        validation_steps = max(1, (self.total_rows - train_size) // self.batch_size)

        # Manually split the dataset
        val_dataset = self.train_dataset.take(validation_steps)

        self.model.summary()

        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=0.0001, verbose=1
            ),
        ]

        # Train with reduced steps to avoid memory issues
        print(
            f"Training model for {epochs} epochs with batch size {self.batch_size}..."
        )
        history = self.model.fit(
            self.train_dataset.skip(validation_steps),
            epochs=epochs,
            steps_per_epoch=min(train_size // self.batch_size, 200),
            validation_data=val_dataset,
            validation_steps=min(validation_steps, 50),
            callbacks=callbacks,
            verbose=1,
        )

        # Simple export
        self.export_model()

        return self.model, history

    def predict(self, X):
        """Convert categorical predictions back to original float values"""
        # Ensure proper alignment
        X = np.ascontiguousarray(X, dtype=np.uint8)

        # Get raw predictions
        raw_preds = self.model.predict(X)

        return raw_preds

    def validate_model(self):
        self.validation_dataset = pd.read_parquet(
            self.data_path_val,
            columns=["era", "data_type", "target"] + self._feature_set,
        )
        validation = self.validation_dataset
        [self.validation_dataset["data_type"] == "validation"]
        del self.validation_dataset["data_type"]

        # Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
        validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

        validation = validation[
            ~validation["era"].isin(
                [
                    str(era).zfill(4)
                    for era in [
                        int(self.train_dataset["era"].unique()[-1]) + i
                        for i in range(4)
                    ]
                ]
            )
        ]

        # Generate predictions against the out-of-sample validation features
        validation["prediction"] = self.model.predict(
            validation[self._feature_set]
        ).squeeze()

        self._validation = validation

        return validation

    def performance_eval(self):

        if self._validation is None:
            print("Please run validation before evaluating the performance!")
            return None

        self._validation["meta_model"] = pd.read_parquet(self.data_path_meta_model)[
            "numerai_meta_model"
        ]

        validation = self._validation

        # Compute the per-era corr between our predictions and the target values
        per_era_corr = validation.groupby("era").apply(
            lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
        )

        # Compute the per-era mmc between our predictions, the meta model, and the target values
        per_era_mmc = (
            validation.dropna()
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
        corr_sharpe = corr_mean / corr_std
        corr_max_drawdown = (
            per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()
        ).max()

        mmc_mean = per_era_mmc.mean()
        mmc_std = per_era_mmc.std(ddof=0)
        mmc_sharpe = mmc_mean / mmc_std
        mmc_max_drawdown = (
            per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()
        ).max()

        pd.DataFrame(
            {
                "mean": [corr_mean, mmc_mean],
                "std": [corr_std, mmc_std],
                "sharpe": [corr_sharpe, mmc_sharpe],
                "max_drawdown": [corr_max_drawdown, mmc_max_drawdown],
            },
            index=["CORR", "MMC"],
        ).T
