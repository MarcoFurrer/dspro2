import os
import json
import gc
import time
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from numerapi import NumerAPI as napi
from numerai_tools.scoring import numerai_corr, correlation_contribution
from typing import Literal


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
        model_id: str = None,
        model_name: str = None,
    ):
        self.path_train = path_train
        self.path_val = path_val
        self.path_meta_model = path_meta_model
        self.path_features = path_features
        self.feature_set = json.load(open(self.path_features, "r", encoding="utf-8"))
        self.output_path = output_path
        self.batch_size = batch_size
        self.subset_features = subset_features
        self.feature_count = None
        self.model = model
        self.model_id = model_id or f"model_{int(time.time())}"
        self.model_name = model_name or f"Model_{self.model_id}"
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

    def _get_parquet_row_count(self, path: str) -> int:
        """Get total row count from parquet file metadata"""
        parquet_file = pq.ParquetFile(path)
        return parquet_file.metadata.num_rows

    def _create_batch_generator(self, parquet_path: str, feature_names: list, 
                               validation_filter: bool = False):
        """Create generator that yields batches from parquet file"""
        def generator():
            parquet_file = pq.ParquetFile(parquet_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df_batch = batch.to_pandas()
                
                if validation_filter and 'data_type' in df_batch.columns:
                    df_batch = df_batch[df_batch['data_type'] == 'validation']
                    if len(df_batch) == 0:
                        continue
                
                X_batch = df_batch[feature_names].values.astype(np.float32)
                y_batch = df_batch['target'].values.astype(np.float32)
                
                del df_batch
                gc.collect()
                
                yield X_batch, y_batch
                
        return generator

    def _create_training_dataset(self, feature_names: list) -> tf.data.Dataset:
        """Create memory-efficient training dataset"""
        output_signature = (
            tf.TensorSpec(shape=(None, len(feature_names)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
        
        return (tf.data.Dataset
                .from_generator(
                    self._create_batch_generator(self.path_train, feature_names),
                    output_signature=output_signature
                )
                .shuffle(buffer_size=1000)
                .prefetch(tf.data.AUTOTUNE))

    def _create_validation_dataset(self, feature_names: list) -> tf.data.Dataset:
        """Create memory-efficient validation dataset"""
        output_signature = (
            tf.TensorSpec(shape=(None, len(feature_names)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
        
        return (tf.data.Dataset
                .from_generator(
                    self._create_batch_generator(self.path_val, feature_names, validation_filter=True),
                    output_signature=output_signature
                )
                .prefetch(tf.data.AUTOTUNE))

    def plot_data(self):
        """Plot data statistics using sample batches"""
        # Load small sample for plotting
        sample_df = pd.read_parquet(self.path_train, nrows=10000)
        
        # Plot the number of rows per era
        sample_df.groupby("era").size().plot(
            title="Number of rows per era (sample)", figsize=(5, 3), xlabel="Era"
        )

        # Plot density histogram of the target
        sample_df["target"].plot(
            kind="hist",
            title="Target Distribution",
            figsize=(5, 3),
            xlabel="Value",
            density=True,
            bins=50,
        )

        del sample_df
        gc.collect()

    def export_model(self):
        """Export model with ID-based naming"""
        model_filename = f"{self.model_id}.keras"
        model_path = os.path.join(self.output_path, model_filename)
        self.model.save(model_path)
        print(f"Model {self.model_name} (ID: {self.model_id}) saved to: {model_path}")
        return model_path

    def train(self, validation_split=0.1, epochs=5):
        """Train the model using memory-efficient batch processing"""
        
        feature_names = self.feature_set["feature_sets"][self.subset_features]
        print(f"Using {len(feature_names)} features for training")
        
        # Create datasets using tf.data pipeline
        train_dataset = self._create_training_dataset(feature_names)
        val_dataset = self._create_validation_dataset(feature_names)
        
        # Calculate steps per epoch from parquet metadata
        total_rows = self._get_parquet_row_count(self.path_train)
        steps_per_epoch = max(1, total_rows // self.batch_size)
        validation_steps = max(1, steps_per_epoch // 10)  # 10% of training steps
        
        print(f"Training samples: {total_rows:,}")
        print(f"Steps per epoch: {steps_per_epoch:,}")
        print(f"Validation steps: {validation_steps:,}")

        self.model.summary()

        # Train the model with tf.data datasets
        print(f"Training model for {epochs} epochs with batch size {self.batch_size}...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
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

        # Export model
        self.export_model()

        return self.model, history

    def predict(self, X):
        """Generate continuous predictions for regression"""
        # Ensure proper data type for continuous features
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Get raw predictions
        raw_preds = self.model.predict(X, batch_size=self.batch_size)

        return raw_preds.squeeze()

    def validate_model(self):
        """Validate the model using memory-efficient batch processing"""
        print("Preparing validation data...")

        feature_names = self.feature_set["feature_sets"][self.subset_features]
        predictions = []
        targets = []
        eras = []

        # Process validation data in batches
        parquet_file = pq.ParquetFile(self.path_val)
        
        for batch in parquet_file.iter_batches(batch_size=self.batch_size * 4):
            df_batch = batch.to_pandas()
            
            # Filter for validation data type
            if 'data_type' in df_batch.columns:
                df_batch = df_batch[df_batch['data_type'] == 'validation']
                
            if len(df_batch) == 0:
                continue
                
            # Downsample eras (every 4th era)
            unique_eras = df_batch['era'].unique()
            selected_eras = unique_eras[::4]
            df_batch = df_batch[df_batch['era'].isin(selected_eras)]
            
            if len(df_batch) == 0:
                continue
            
            # Extract features and make predictions
            X_batch = df_batch[feature_names].values.astype(np.float32)
            batch_predictions = self.model.predict(X_batch, verbose=0)
            
            # Collect results
            predictions.extend(batch_predictions.squeeze())
            targets.extend(df_batch['target'].values)
            eras.extend(df_batch['era'].values)
            
            del df_batch, X_batch, batch_predictions
            gc.collect()

        # Create validation results
        validation_results = pd.DataFrame({
            'era': eras,
            'target': targets,
            'prediction': predictions
        })
        
        self._validation = validation_results
        print(f"Validation completed with {len(validation_results):,} samples")

        return validation_results

    def performance_eval(self):
        """Evaluate model performance using validation results"""
        if self._validation is None:
            print("Please run validation before evaluating the performance!")
            return None

        # Compute the per-era correlation between predictions and target values
        per_era_corr = (
            self._validation
            .groupby("era")
            .apply(
                lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
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

        # Plot the cumulative per-era correlation
        per_era_corr.cumsum().plot(
            title="Cumulative Validation CORR",
            kind="line",
            figsize=(8, 4),
            legend=False,
        )

        # Compute performance metrics
        corr_mean = per_era_corr.mean()
        corr_std = per_era_corr.std(ddof=0)
        sharpe_ratio = corr_mean / corr_std if corr_std > 0 else 0
        max_drawdown = (
            per_era_corr.cumsum().expanding(min_periods=1).max()
            - per_era_corr.cumsum()
        ).max()

        performance_metrics = pd.DataFrame({
            "mean": [corr_mean],
            "std": [corr_std],
            "sharpe": [sharpe_ratio],
            "max_drawdown": [max_drawdown],
        }, index=["CORR"]).T

        print("\nPerformance Metrics:")
        print(performance_metrics)
        
        return performance_metrics
