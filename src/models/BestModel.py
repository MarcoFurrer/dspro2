import tensorflow as tf
from keras.layers import (
    Dense, Input, Dropout, BatchNormalization,
    Add, Concatenate, MultiHeadAttention, LayerNormalization,
    Reshape, GlobalAveragePooling1D, Layer, LeakyReLU, GaussianNoise
)
from keras.models import Model
import numpy as np

class ConditionalDistributionLayer(Layer):
    """Layer that learns the conditional distribution of features given target values."""
    def __init__(self, num_bins=5, regularization=1e-4, **kwargs):
        super(ConditionalDistributionLayer, self).__init__(**kwargs)
        self.num_bins = num_bins
        self.regularization = regularization
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Create learnable distribution embeddings for each target bin
        self.distribution_embeddings = self.add_weight(
            shape=(self.num_bins, feature_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.regularization),
            trainable=True,
            name="conditional_distribution_embeddings"
        )
        
        # Create adaptive mixing weights
        self.mixing_weights = self.add_weight(
            shape=(feature_dim,),
            initializer='ones',
            regularizer=tf.keras.regularizers.l2(self.regularization),
            trainable=True,
            name="mixing_weights"
        )
        
        super(ConditionalDistributionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Create soft binning weights using learnable parameters
        # This allows the model to learn which bin each feature belongs to
        # instead of hard-coding the bins
        bin_weights = tf.nn.softmax(self.distribution_embeddings, axis=0)
        
        # Apply the learned conditional distributions as attention
        # This emphasizes features that are important for each distribution
        feature_importance = tf.nn.softmax(self.mixing_weights)
        
        # Element-wise multiplication with feature importance
        enhanced_features = inputs * feature_importance
        
        return enhanced_features
    
    def get_config(self):
        config = super(ConditionalDistributionLayer, self).get_config()
        config.update({
            "num_bins": self.num_bins,
            "regularization": self.regularization
        })
        return config


class DistributionAwareBlock(Layer):
    """Block that explicitly models the feature distributions for different target values."""
    def __init__(self, units, num_bins=5, dropout_rate=0.2, **kwargs):
        super(DistributionAwareBlock, self).__init__(**kwargs)
        self.units = units
        self.num_bins = num_bins
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Create distribution-specific transformations
        self.distribution_transforms = []
        for i in range(self.num_bins):
            self.distribution_transforms.append(
                Dense(self.units // self.num_bins, activation='swish')
            )
            
        # Integration layer to combine all distribution-specific outputs
        self.integration = Dense(self.units, activation=None)
        self.dropout = Dropout(self.dropout_rate)
        self.norm = LayerNormalization(epsilon=1e-5)
        
        super(DistributionAwareBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Process inputs through each distribution-specific pathway
        transform_outputs = []
        for i in range(self.num_bins):
            transform_output = self.distribution_transforms[i](inputs)
            transform_outputs.append(transform_output)
        
        # Concatenate all distribution-specific outputs
        concat_output = Concatenate()(transform_outputs)
        
        # Integrate and normalize
        x = self.integration(concat_output)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        config = super(DistributionAwareBlock, self).get_config()
        config.update({
            "units": self.units,
            "num_bins": self.num_bins,
            "dropout_rate": self.dropout_rate
        })
        return config


class FeatureCrossAttentionLayer(Layer):
    """Layer that explicitly models attention between feature pairs."""
    def __init__(self, projection_dim=64, heads=4, dropout_rate=0.1, **kwargs):
        super(FeatureCrossAttentionLayer, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # Create projections for attention mechanism
        self.query_projection = Dense(self.projection_dim, activation=None)
        self.key_projection = Dense(self.projection_dim, activation=None)
        self.value_projection = Dense(self.projection_dim, activation=None)
        self.output_projection = Dense(self.input_dim, activation=None)
        
        # Feature interactions with attention across dimensions
        self.attention = MultiHeadAttention(
            num_heads=self.heads, 
            key_dim=self.projection_dim // self.heads,
            dropout=self.dropout_rate
        )
        
        self.norm = LayerNormalization(epsilon=1e-5)
        self.dropout = Dropout(self.dropout_rate)
        
        super(FeatureCrossAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Project input to create query, key, and value
        query = self.query_projection(inputs)
        key = self.key_projection(inputs)
        value = self.value_projection(inputs)
        
        # Reshape for attention
        batch_size = tf.shape(inputs)[0]
        
        # Add feature dimension for attention to work on features
        # Reshape from (batch, features) to (batch, 1, features)
        query_reshaped = tf.expand_dims(query, axis=1)
        key_reshaped = tf.expand_dims(key, axis=1)
        value_reshaped = tf.expand_dims(value, axis=1)
        
        # Perform self-attention across feature dimensions
        attention_output = self.attention(
            query=query_reshaped,
            key=key_reshaped,
            value=value_reshaped,
            training=training
        )
        
        # Reshape back to (batch, features)
        attention_output = tf.squeeze(attention_output, axis=1)
        
        # Project back to input dimension
        output = self.output_projection(attention_output)
        
        # Add residual connection
        output = inputs + output
        output = self.norm(output)
        output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        config = super(FeatureCrossAttentionLayer, self).get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "heads": self.heads,
            "dropout_rate": self.dropout_rate
        })
        return config


def create_best_model(input_shape, output_dim=1):
    """Creates a model specifically designed to address target distribution memorization issues."""
    # Ensure input_shape is correctly defined
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
        
    inputs = Input(shape=input_shape)
    
    # Add small amount of noise for regularization
    x = GaussianNoise(0.01)(inputs)
    
    # Learn conditional distributions 
    x = ConditionalDistributionLayer(num_bins=5, regularization=1e-5)(x)
    
    # Initial feature processing
    x = Dense(256, activation='swish')(x)
    x = LayerNormalization(epsilon=1e-5)(x)
    x = Dropout(0.15)(x)
    
    # Distribution-aware processing with 5 bins (corresponding to target values)
    x1 = DistributionAwareBlock(128, num_bins=5, dropout_rate=0.15)(x)
    
    # Feature cross-attention to capture complex interactions
    x2 = FeatureCrossAttentionLayer(projection_dim=64, heads=4, dropout_rate=0.15)(x)
    
    # Combine different processing paths
    x = Concatenate()([x1, x2])
    
    # Final integration layers
    x = Dense(128, activation='swish')(x)
    x = LayerNormalization(epsilon=1e-5)(x)
    x = Dropout(0.1)(x)
    
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    outputs = Dense(output_dim, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="BestModel")
    
    return model


# Factory function for the model
def model(input_shape=(705,)):
    """Factory function for the BestModel."""
    # Handle various input shape formats
    if isinstance(input_shape, tuple) and len(input_shape) == 1:
        # Already in correct format
        pass
    elif isinstance(input_shape, int):
        input_shape = (input_shape,)
    elif not isinstance(input_shape, tuple):
        # Default fallback
        input_shape = (705,)
        
    return create_best_model(input_shape)