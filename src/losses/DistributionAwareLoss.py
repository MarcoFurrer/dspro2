import tensorflow as tf
import numpy as np

class DistributionAwareLoss(tf.keras.losses.Loss):
    """
    Custom loss function that penalizes the model for just learning the target distribution.
    
    This loss combines:
    1. Mean Absolute Error for basic regression accuracy
    2. A penalty term if predictions match the overall target distribution too closely
    3. A conditional correlation term that encourages learning true feature-target relationships
    """
    def __init__(self, distribution_penalty_weight=0.3, correlation_reward_weight=0.2, name="distribution_aware_loss", **kwargs):
        super(DistributionAwareLoss, self).__init__(name=name, **kwargs)
        self.distribution_penalty_weight = distribution_penalty_weight
        self.correlation_reward_weight = correlation_reward_weight
        
        # Initialize moving statistics for target distribution
        self.target_mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.target_var = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.decay = 0.99  # Decay rate for moving statistics
        
    def call(self, y_true, y_pred):
        # Base MAE loss
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Update target statistics with moving average
        batch_mean = tf.reduce_mean(y_true)
        batch_var = tf.math.reduce_variance(y_true)
        
        self.target_mean.assign(self.decay * self.target_mean + (1 - self.decay) * batch_mean)
        self.target_var.assign(self.decay * self.target_var + (1 - self.decay) * batch_var)
        
        # Calculate prediction distribution statistics
        pred_mean = tf.reduce_mean(y_pred)
        pred_var = tf.math.reduce_variance(y_pred)
        
        # Distribution matching penalty - penalize if prediction distribution is too similar to target
        # This prevents the model from just memorizing the overall distribution
        mean_diff = tf.square(batch_mean - pred_mean)
        var_ratio = tf.maximum(batch_var / (pred_var + 1e-8), pred_var / (batch_var + 1e-8))
        
        distribution_penalty = mean_diff + tf.maximum(0.0, 1.0 - var_ratio)
        
        # Calculate correlation between predictions and targets
        # Higher correlation is better, so we subtract from loss
        # Use Spearman-like approach by looking at ranks
        y_true_ranks = tf.argsort(tf.argsort(y_true, axis=0), axis=0)
        y_pred_ranks = tf.argsort(tf.argsort(y_pred, axis=0), axis=0)
        
        # Convert to float and normalize
        y_true_ranks = tf.cast(y_true_ranks, tf.float32) / tf.maximum(1.0, tf.cast(tf.shape(y_true)[0] - 1, tf.float32))
        y_pred_ranks = tf.cast(y_pred_ranks, tf.float32) / tf.maximum(1.0, tf.cast(tf.shape(y_pred)[0] - 1, tf.float32))
        
        # Calculate rank correlation
        rank_correlation = tf.reduce_mean(tf.square(y_true_ranks - y_pred_ranks))
        correlation_reward = 1.0 - rank_correlation  # Higher is better
        
        # Combine all components
        total_loss = mae_loss + \
                    self.distribution_penalty_weight * distribution_penalty - \
                    self.correlation_reward_weight * correlation_reward
        
        return total_loss


class FeatureDistributionLoss(tf.keras.losses.Loss):
    """
    A loss function that explicitly encourages the model to learn feature-conditional distributions.
    
    For each target value bin, it learns what the expected feature distributions should be
    and penalizes deviations from these learned distributions.
    """
    def __init__(self, num_bins=5, bin_smoothing=0.1, name="feature_distribution_loss", **kwargs):
        super(FeatureDistributionLoss, self).__init__(name=name, **kwargs)
        self.num_bins = num_bins
        self.bin_smoothing = bin_smoothing
        self.initialized = False
        
    def build(self, input_shape):
        # Will be initialized on first call
        self.initialized = False
        
    def call(self, y_true, y_pred):
        # Standard MAE as base loss
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Bin the targets for conditional distribution learning
        # Create bins from 0 to 1 with num_bins divisions
        bin_edges = tf.linspace(0.0, 1.0, self.num_bins + 1)
        bin_width = 1.0 / self.num_bins
        
        # Create soft bin assignments with smoothing
        # This allows for gradient flow and reduces discretization issues
        soft_bin_assignments = []
        for i in range(self.num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            # Calculate distance from bin center
            bin_center = (lower + upper) / 2.0
            distance = tf.abs(y_true - bin_center) / bin_width
            # Create soft assignment using Gaussian-like function
            soft_assignment = tf.exp(-tf.square(distance) / self.bin_smoothing)
            soft_bin_assignments.append(soft_assignment)
            
        # Stack and normalize
        soft_bin_assignments = tf.stack(soft_bin_assignments, axis=1)
        soft_bin_assignments = soft_bin_assignments / (tf.reduce_sum(soft_bin_assignments, axis=1, keepdims=True) + 1e-8)
        
        # Create similar bin assignments for predictions
        pred_bin_assignments = []
        for i in range(self.num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            bin_center = (lower + upper) / 2.0
            distance = tf.abs(y_pred - bin_center) / bin_width
            soft_assignment = tf.exp(-tf.square(distance) / self.bin_smoothing)
            pred_bin_assignments.append(soft_assignment)
            
        pred_bin_assignments = tf.stack(pred_bin_assignments, axis=1)
        pred_bin_assignments = pred_bin_assignments / (tf.reduce_sum(pred_bin_assignments, axis=1, keepdims=True) + 1e-8)
        
        # Calculate KL-divergence between true and predicted bin assignments
        bin_kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                soft_bin_assignments * tf.math.log(soft_bin_assignments / (pred_bin_assignments + 1e-8) + 1e-8), 
                axis=1
            )
        )
        
        # Total loss is MAE plus bin distribution matching
        total_loss = mae_loss + 0.2 * bin_kl_loss
        
        return total_loss


def get_best_loss():
    """Factory function to get the best loss function for our model."""
    return DistributionAwareLoss(
        distribution_penalty_weight=0.3,
        correlation_reward_weight=0.2
    )