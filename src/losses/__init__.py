"""
Loss functions for the project.

This module contains various loss functions including:
- DistributionAwareLoss: A loss function that penalizes target distribution memorization
- FeatureDistributionLoss: A loss function that encourages learning feature-conditional distributions
"""

from .DistributionAwareLoss import (
    DistributionAwareLoss,
    FeatureDistributionLoss,
    get_best_loss
)

__all__ = [
    'DistributionAwareLoss', 
    'FeatureDistributionLoss',
    'get_best_loss'
]