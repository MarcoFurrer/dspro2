class Visualizer:
    """Class for visualization of model performance and data"""
    
    def __init__(self):
        """Initialize visualization settings"""
        import matplotlib.pyplot as plt
        self.plt = plt
        
    def plot_validation_metrics(self, per_era_corr, per_era_mmc=None):
        """Plot validation metrics by era and cumulatively"""
        # Plot the per-era correlation
        self.plt.figure(figsize=(8, 4))
        per_era_corr.plot(
            title="Validation CORR",
            kind="bar",
            figsize=(8, 4),
            xticks=[],
            legend=False,
            snap=False
        )
        self.plt.show()
        
        if per_era_mmc is not None and not per_era_mmc.empty:
            self.plt.figure(figsize=(8, 4))
            per_era_mmc.plot(
                title="Validation MMC",
                kind="bar",
                figsize=(8, 4),
                xticks=[],
                legend=False,
                snap=False
            )
            self.plt.show()
        
        # Plot the cumulative per-era correlation
        self.plt.figure(figsize=(8, 4))
        per_era_corr.cumsum().plot(
            title="Cumulative Validation CORR",
            kind="line",
            figsize=(8, 4),
            legend=False
        )
        self.plt.show()
        
        if per_era_mmc is not None and not per_era_mmc.empty:
            self.plt.figure(figsize=(8, 4))
            per_era_mmc.cumsum().plot(
                title="Cumulative Validation MMC",
                kind="line",
                figsize=(8, 4),
                legend=False
            )
            self.plt.show()
            
    def plot_training_history(self, history):
        """Plot training and validation loss/metrics"""
        self.plt.figure(figsize=(12, 4))
        
        # Plot loss
        self.plt.subplot(1, 2, 1)
        self.plt.plot(history.history['loss'], label='Training Loss')
        self.plt.plot(history.history['val_loss'], label='Validation Loss')
        self.plt.title('Model Loss')
        self.plt.xlabel('Epoch')
        self.plt.ylabel('Loss')
        self.plt.legend()
        
        # Plot MAE if available
        if 'mae' in history.history:
            self.plt.subplot(1, 2, 2)
            self.plt.plot(history.history['mae'], label='Training MAE')
            self.plt.plot(history.history['val_mae'], label='Validation MAE')
            self.plt.title('Model MAE')
            self.plt.xlabel('Epoch')
            self.plt.ylabel('MAE')
            self.plt.legend()
            
        self.plt.tight_layout()
        self.plt.show()
        
    def plot_data_distribution(self, train_data, feature_subset=None, target_col='target'):
        """Plot distributions of features and targets in the dataset"""
        import pandas as pd
        
        # Plot density histogram of the target
        self.plt.figure(figsize=(8, 4))
        train_data[target_col].plot(
            kind="hist",
            title="Target Distribution",
            figsize=(8, 4),
            xlabel="Value",
            density=True,
            bins=50
        )
        self.plt.show()
        
        # If feature subset provided, plot some feature distributions
        if feature_subset is not None:
            features_to_plot = feature_subset[:5] if len(feature_subset) > 5 else feature_subset
            
            self.plt.figure(figsize=(15, 3*len(features_to_plot)))
            for i, feature in enumerate(features_to_plot):
                self.plt.subplot(len(features_to_plot), 1, i+1)
                train_data[feature].plot(
                    kind="hist",
                    title=f"Feature {feature} Distribution",
                    density=True,
                    bins=50
                )
            self.plt.tight_layout()
            self.plt.show()
            
    def plot_feature_importance(self, feature_names, importance_values):
        """Plot feature importance if available"""
        import pandas as pd
        
        # Sort features by importance
        indices = pd.Series(importance_values, index=feature_names).sort_values(ascending=False)
        
        # Plot top 20 features
        top_n = min(20, len(indices))
        self.plt.figure(figsize=(10, 8))
        indices[:top_n].plot.bar(colormap='viridis')
        self.plt.title('Feature Importance')
        self.plt.tight_layout()
        self.plt.show()
        
    def plot_prediction_vs_actual(self, predictions, actual):
        """Plot predicted vs actual values with a regression line"""
        import numpy as np
        
        # Create scatter plot
        self.plt.figure(figsize=(8, 8))
        self.plt.scatter(actual, predictions, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(actual, predictions, 1)
        p = np.poly1d(z)
        self.plt.plot(actual, p(actual), "r--")
        
        self.plt.xlabel('Actual Values')
        self.plt.ylabel('Predicted Values')
        self.plt.title('Predictions vs Actual Values')
        self.plt.axis('equal')
        self.plt.tight_layout()
        self.plt.show()