import tensorflow as tf
from keras.layers import (
    Dense, Input, Dropout, BatchNormalization,
    Add, Concatenate, MultiHeadAttention, LayerNormalization,
    Reshape, GlobalAveragePooling1D, Layer
)
from keras.models import Model
import numpy as np

class FeatureInteractionBlock(Layer):
    """Enhanced feature interaction block that captures both linear and non-linear relationships"""
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(FeatureInteractionBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Linear path
        self.linear = Dense(self.units, activation=None)
        
        # Deep path with swish activation for better gradients
        self.deep1 = Dense(self.units, activation='swish')
        self.deep2 = Dense(self.units, activation='swish')
        
        # Cross-feature path to model interactions
        self.cross_w = self.add_weight(
            shape=(input_dim, input_dim),
            initializer='glorot_uniform',
            name='cross_weights',
            regularizer=tf.keras.regularizers.l2(1e-5)
        )
        self.cross_b = self.add_weight(
            shape=(input_dim,),
            initializer='zeros',
            name='cross_bias'
        )
        
        # Cross-feature projection
        self.cross_projection = Dense(self.units, activation=None)
        
        # Normalization and regularization
        self.batch_norm = BatchNormalization()
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        
        # Output integration
        self.output_layer = Dense(self.units, activation='swish')
        
    def call(self, inputs, training=None):
        # Linear path
        linear_out = self.linear(inputs)
        
        # Deep path
        deep_out = self.deep1(inputs)
        deep_out = self.dropout(deep_out, training=training)
        deep_out = self.deep2(deep_out)
        
        # Cross-feature path: x_0 * (x_l^T * W + b)
        # Modified implementation to avoid shape issues
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]
        
        # Reshape inputs for batch-wise operations
        x_flat = tf.reshape(inputs, [batch_size, input_dim])
        
        # Compute cross terms with explicit shapes
        cross_term = tf.matmul(x_flat, self.cross_w) + self.cross_b
        cross_out = self.cross_projection(cross_term)
        
        # Combine all paths with fixed dimensions
        combined = Concatenate()([linear_out, deep_out, cross_out])
        combined = self.batch_norm(combined, training=training)
        
        # Output integration
        output = self.output_layer(combined)
        output = self.layer_norm(output)
        output = self.dropout(output, training=training)
        
        return output
        
    def get_config(self):
        config = super(FeatureInteractionBlock, self).get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

class SelfAttentionBlock(Layer):
    """Self-attention block to capture global feature dependencies"""
    def __init__(self, num_heads=4, key_dim=16, dropout_rate=0.1, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # Reshape to sequence format for attention
        self.reshape_in = Reshape((1, self.input_dim))
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        
        # Normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        # FFN using sequential to avoid shape issues
        self.ffn = tf.keras.Sequential([
            Dense(self.input_dim, activation='swish'),  # Reduced size to avoid dimension mismatch
            Dropout(self.dropout_rate),
            Dense(self.input_dim)
        ])
        
        # Reshape back
        self.reshape_out = Reshape((self.input_dim,))
        
    def call(self, inputs, training=None):
        # Reshape for sample-wise attention
        x = self.reshape_in(inputs)
        
        # Self attention
        attn_output = self.mha(x, x, x, training=training)
        x1 = self.norm1(x + attn_output)
        
        # FFN
        ffn_output = self.ffn(x1, training=training)
        x2 = self.norm2(x1 + ffn_output)
        
        # Reshape back
        output = self.reshape_out(x2)
        
        return output
    
    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

class ResidualBlock(Layer):
    """Residual block with pre-activation"""
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Normalization and dense layers
        self.norm1 = BatchNormalization()
        self.dense1 = Dense(self.units, activation='swish')
        self.dropout1 = Dropout(self.dropout_rate)
        
        self.norm2 = BatchNormalization()
        self.dense2 = Dense(self.units, activation=None)
        self.dropout2 = Dropout(self.dropout_rate)
        
        # Projection if needed
        self.projection = None
        if input_dim != self.units:
            self.projection = Dense(self.units, activation=None)
        
    def call(self, inputs, training=None):
        # Pre-activation
        x = self.norm1(inputs, training=training)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        
        x = self.norm2(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        # Projection for residual connection if needed
        if self.projection:
            shortcut = self.projection(inputs)
        else:
            shortcut = inputs
            
        # Add residual connection
        output = Add()([shortcut, x])
        
        return output
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

def create_correlation_model(input_shape, output_dim=1):
    """Creates a model specifically designed to learn feature correlations rather than just distribution"""
    # Ensure input_shape is correctly defined
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
        
    inputs = Input(shape=input_shape)
    
    # First extract basic features with residual blocks
    x = Dense(128, activation='swish')(inputs)
    x = BatchNormalization()(x)
    
    # Multiple residual blocks for deeper representation
    x = ResidualBlock(128, dropout_rate=0.2)(x)
    x = ResidualBlock(128, dropout_rate=0.2)(x)
    
    # Feature interaction block to explicitly model pairwise interactions
    x = FeatureInteractionBlock(128, dropout_rate=0.2)(x)  # Reduced size to avoid dimension mismatch
    
    # Self-attention to capture global dependencies
    attn = SelfAttentionBlock(num_heads=4, key_dim=16, dropout_rate=0.1)(x)
    
    # Combine with a skip connection
    x = Concatenate()([x, attn])
    
    # Final layers
    x = Dense(64, activation='swish')(x)  # Reduced size for better convergence
    x = Dropout(0.1)(x)  # Reduced dropout rate
    
    # Output layer
    outputs = Dense(output_dim, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="CorrelationModel")
    
    return model

# We need a function that returns a model (not an instance)
def model(input_shape=(705,)):
    """Factory function to create a model with specified input shape"""
    # Handle various input shape formats
    if isinstance(input_shape, tuple) and len(input_shape) == 1:
        # Already in correct format
        pass
    elif isinstance(input_shape, int):
        input_shape = (input_shape,)
    elif not isinstance(input_shape, tuple):
        # Default fallback
        input_shape = (705,)
        
    return create_correlation_model(input_shape)