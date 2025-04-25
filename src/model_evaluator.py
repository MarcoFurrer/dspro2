import pandas as pd
import numpy as np
from numerai_tools.scoring import numerai_corr, correlation_contribution

class ModelEvaluator:
    def __init__(self, data_path_meta_model=None):
        self.data_path_meta_model = data_path_meta_model
        self._validation = None
    
    def validate_model(self, model, validation_data, feature_set):
        """Validate model on validation dataset and return performance metrics"""
        # Generate predictions against the out-of-sample validation features
        validation = validation_data.copy()
        validation["prediction"] = model.predict(validation[feature_set])
        
        self._validation = validation
        return validation
        
    def performance_eval(self):
        """Calculate and return performance metrics based on validation results"""
        if self._validation is None:
            raise ValueError("Please run validation before evaluating the performance!")
        
        # Load meta model data if available
        if self.data_path_meta_model:
            self._validation["meta_model"] = pd.read_parquet(
                self.data_path_meta_model
            )["numerai_meta_model"]
        else:
            print("No meta model data path provided. MMC metrics will not be available.")
            self._validation["meta_model"] = np.nan
            
        validation = self._validation

        # Compute the per-era corr between our predictions and the target values
        per_era_corr = validation.groupby("era").apply(
            lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
        )
        
        # If meta model data is available, compute MMC
        if self.data_path_meta_model:
            per_era_mmc = validation.dropna().groupby("era").apply(
                lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
            )
        else:
            per_era_mmc = pd.Series(index=per_era_corr.index, dtype='float64')
        
        # Compute performance metrics
        corr_mean = per_era_corr.mean()
        corr_std = per_era_corr.std(ddof=0)
        corr_sharpe = corr_mean / corr_std if corr_std > 0 else 0
        corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()
        
        mmc_mean = per_era_mmc.mean()
        mmc_std = per_era_mmc.std(ddof=0) 
        mmc_sharpe = mmc_mean / mmc_std if mmc_std > 0 else 0
        mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max() if not per_era_mmc.empty else 0
        
        metrics = pd.DataFrame({
            "mean": [corr_mean, mmc_mean],
            "std": [corr_std, mmc_std],
            "sharpe": [corr_sharpe, mmc_sharpe],
            "max_drawdown": [corr_max_drawdown, mmc_max_drawdown]
        }, index=["CORR", "MMC"])
        
        results = {
            "metrics": metrics,
            "per_era_corr": per_era_corr,
            "per_era_mmc": per_era_mmc
        }
        
        return results