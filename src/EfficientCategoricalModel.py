import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hides GPU from TensorFlow
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"  # disables broken XLA

from src.data_handler import DataHandler
from src.model_manager import ModelManager
from src.model_evaluator import ModelEvaluator
from src.Visualizer import Visualizer

class EfficientCategoricalModel:
    """
    Main class that orchestrates the workflow of data loading,
    model training, evaluation and visualization.
    """
    def __init__(self, 
                 data_path_train=None, 
                 data_path_val=None, 
                 data_path_metadata=None,
                 data_path_meta_model=None, 
                 output_path='exports', 
                 batch_size=64, 
                 subset_features="small",
                 model=None,
                 optimizer=None):
        
        # Initialize the component classes
        self.data_handler = DataHandler(
            data_path_train=data_path_train, 
            data_path_val=data_path_val, 
            data_path_metadata=data_path_metadata,
            batch_size=batch_size,
            subset_features=subset_features
        )
        
        self.model_manager = ModelManager(
            feature_count=None,  # Will be set after data is loaded
            output_path=output_path,
            model=model,
            optimizer=optimizer
        )
        
        self.evaluator = ModelEvaluator(
            data_path_meta_model=data_path_meta_model
        )
        
        self.visualizer = Visualizer()
    
    def train(self, validation_split=0.1, epochs=5):
        """Train the model using memory-efficient batch processing"""
        # Download data if needed and get dataset info
        if self.data_handler.data_path_train is None or self.data_handler.data_path_metadata is None:
            self.data_handler.download_data()
        
        # Get dataset info and feature set
        self.data_handler.get_dataset_info()
        
        # Update model manager with feature count
        self.model_manager.feature_count = self.data_handler.feature_count
        
        # Create full dataset pipeline
        print("Creating dataset pipeline...")
        full_dataset = self.data_handler.create_dataset_pipeline(self.data_handler.data_path_train)
        
        # Calculate dataset sizes and steps
        train_size = int(self.data_handler.total_rows * (1 - validation_split))
        steps_per_epoch = train_size // self.data_handler.batch_size
        validation_steps = max(1, (self.data_handler.total_rows - train_size) // self.data_handler.batch_size)
        
        # Manually split the dataset
        val_dataset = full_dataset.take(validation_steps)
        train_dataset = full_dataset.skip(validation_steps)
        
        # Train the model
        model, history = self.model_manager.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs
        )
        
        # Visualize training history
        self.visualizer.plot_training_history(history)
        
        return model, history
    
    def validate_model(self, model_filepath=None):
        """Validate the model on the validation dataset"""
        # Load model from file if specified
        if model_filepath is not None:
            self.model_manager.load_model(model_filepath)
        
        # Ensure we have the feature set
        if self.data_handler._feature_set is None:
            self.data_handler.get_dataset_info()
            
        # Load validation data
        validation_data = self.data_handler.load_validation_data()
        
        # Run validation
        validation_results = self.evaluator.validate_model(
            model=self.model_manager,
            validation_data=validation_data,
            feature_set=self.data_handler.feature_set
        )
        
        return validation_results
    
    def performance_eval(self):
        """Evaluate model performance and visualize results"""
        # Calculate performance metrics
        eval_results = self.evaluator.performance_eval()
        
        # Print metrics
        print("Performance Metrics:")
        print(eval_results["metrics"])
        
        # Visualize performance
        self.visualizer.plot_validation_metrics(
            eval_results["per_era_corr"],
            eval_results["per_era_mmc"]
        )
        
        return eval_results["metrics"]
    
    def export_model(self, model_name=None):
        """Export the trained model"""
        return self.model_manager.export_model(model_name=model_name)
    
    @property
    def model(self):
        """Return the trained model"""
        return self.model_manager.model