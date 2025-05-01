"""
Neural network model architectures for categorical data prediction.

This module contains various deep learning architectures including:
- Deep: A complex model with feature interactions and attention mechanisms
- ImprovedModel: Enhanced architecture for better categorical feature handling
- CorrelationModel: Model focused on capturing feature correlations
- BestModel: Model designed to avoid target distribution memorization
"""

# Make models accessible directly from the models package
from .Deep import model as deep_model
from .ImprovedModel import model as improved_model 
from .CorrelationModel import model as correlation_model
from .BestModel import model as best_model

# Custom layers can also be exported for reuse
from .Deep import FeatureInteractionLayer, attention_block

# Ordinal model is included if available
try:
    from .OrdinalModel import model as ordinal_model
    __all__ = ['deep_model', 'improved_model', 'correlation_model', 'ordinal_model', 'best_model',
               'FeatureInteractionLayer', 'attention_block']
except ImportError:
    __all__ = ['deep_model', 'improved_model', 'correlation_model', 'best_model',
               'FeatureInteractionLayer', 'attention_block']