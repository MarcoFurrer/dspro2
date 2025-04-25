import os
import json
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from numerapi import NumerAPI

class DataHandler:
    def __init__(self, 
                data_path_train=None, 
                data_path_val=None, 
                data_path_metadata=None,
                batch_size=64, 
                subset_features="small"):
        
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.data_path_metadata = data_path_metadata
        self.batch_size = batch_size
        self._subset_features = subset_features
        self._feature_set = None
        self.feature_count = None
        self.total_rows = None
        self.feature_cols = None
        self.target_cols = None
        self.data_version = "v5.0"
        self.napi = NumerAPI()
    
    def download_data(self):
        """Download required datasets for training and validation"""
        # List the datasets and available versions
        all_datasets = self.napi.list_datasets()
        dataset_versions = list(set(d.split('/')[0] for d in all_datasets))
        print("Available versions:\n", dataset_versions)
        # Print all files available for download for our version
        current_version_files = [f for f in all_datasets if f.startswith(self.data_version)]
        print("Available", self.data_version, "files:\n", current_version_files)

        # Download the feature metadata file
        self.napi.download_dataset(f"{self.data_version}/features.json")
        self.napi.download_dataset(f"{self.data_version}/validation.parquet")
        self.napi.download_dataset(f"{self.data_version}/train.parquet")

        self.data_path_metadata = f"{self.data_version}/features.json"
        self.data_path_train = f"{self.data_version}/train.parquet"
        self.data_path_val = f"{self.data_version}/validation.parquet"
        
        return self.data_path_train, self.data_path_val, self.data_path_metadata

    def get_dataset_info(self):
        """Get basic information about the dataset"""
        # Read the metadata and display
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
        df_sample = pd.read_parquet(self.data_path_train,
                                  columns=["era", "target"] + self._feature_set)

        df_sample = df_sample.head(1)
        # Identify feature and target columns
        feature_cols = [col for col in df_sample.columns if col.startswith('feature_')]
        target_cols = [col for col in df_sample.columns if col.startswith('target')]
        
        total_rows = parquet_file.metadata.num_rows
        
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.feature_count = len(feature_cols)
        self.total_rows = total_rows
        
        print(f"Dataset has {total_rows:,} rows")
        print(f"Found {self.feature_count} feature columns and {len(target_cols)} target columns")
        
        return self._feature_set, self.feature_count, self.total_rows

    def create_dataset_pipeline(self, data_path, is_training=True):
        """Create an efficient TF dataset pipeline that processes data in batches"""
        # Define dataset batch generator
        def generator():
            parquet_file = pq.ParquetFile(data_path)
            
            # Use smaller read batches to reduce memory pressure
            read_batch_size = min(10000, self.total_rows // 50)
            
            for batch in parquet_file.iter_batches(batch_size=read_batch_size):
                df_batch = batch.to_pandas()
                
                # Extract features and target with explicit alignment
                X_batch = df_batch[self.feature_cols].values
                X_batch = np.ascontiguousarray(X_batch, dtype=np.float32)
                
                # Convert target values to float32 with explicit alignment
                y_batch = df_batch[self.target_cols].values.squeeze()
                y_batch = np.ascontiguousarray(y_batch, dtype=np.float32)
                
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
        
        # Define output shapes and types
        output_signature = (
            tf.TensorSpec(shape=(None, self.feature_count), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Configure dataset for performance - reduce shuffle buffer size
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if is_training:
            # Use a smaller shuffle buffer to avoid memory issues
            dataset = dataset.shuffle(buffer_size=1000)
            
        return dataset
    
    def load_validation_data(self):
        """Load validation data specifically for model evaluation"""
        validation = pd.read_parquet(
            self.data_path_val,
            columns=["era", "data_type", "target"] + self._feature_set
        )
        validation = validation[validation["data_type"] == "validation"]
        del validation["data_type"]
        
        # Get train eras for embargo calculation
        train = pd.read_parquet(
            self.data_path_train,
            columns=["era"]
        )
        
        # Downsample to every 4th era to reduce memory usage
        validation = validation[validation["era"].isin(validation["era"].unique()[::4])]
        
        # Embargo eras to avoid data leakage
        last_train_era = int(train["era"].unique()[-1])
        eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
        validation = validation[~validation["era"].isin(eras_to_embargo)]
        
        return validation
    
    @property
    def feature_set(self):
        return self._feature_set