import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Input, Dropout, BatchNormalization, 
    Add, Activation, MultiHeadAttention, Reshape, 
    Concatenate, Multiply, GlobalAveragePooling1D, Conv1D
)
from tensorflow.keras.models import Model
import numpy as np

class OrdinalLayer(Layer):
    """
    Custom layer for ordinal regression - outputs cumulative probabilities
    for ordinal target values [0, 0.25, 0.5, 0.75, 1.0]
    """
    def __init__(self, num_classes=5, **kwargs):
        self.num_classes = num_classes
        self.supports_masking = True
        super(OrdinalLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create trainable thresholds for ordinal regression
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.num_classes - 1,),
            initializer=tf.keras.initializers.Constant(np.linspace(0.2, 0.8, self.num_classes - 1)),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.05, 0.95)
        )
        super(OrdinalLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        # Calculate probabilities of crossing each threshold
        return tf.sigmoid(50.0 * (inputs - self.thresholds))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes - 1
    
    def get_config(self):
        config = super(OrdinalLayer, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config

# Enhanced feature interaction layer with DCN v2 (Deep & Cross Network)
class CrossNetworkLayer(Layer):
    def __init__(self, projection_dim=None, kernel_initializer='glorot_uniform', **kwargs):
        super(CrossNetworkLayer, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        
        # If projection dimension is specified, use it, otherwise use input dimension
        self.projection_dim = self.projection_dim or input_dim
        
        # Define U, V, and bias
        self.U = self.add_weight(
            name='U',
            shape=(input_dim, self.projection_dim),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        self.V = self.add_weight(
            name='V',
            shape=(input_dim, self.projection_dim),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(input_dim,),
            initializer='zeros',
            trainable=True
        )
        
        super(CrossNetworkLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        x0 = inputs
        x = inputs
        
        # Project x to lower dimension
        x_proj = tf.matmul(x, self.U)
        
        # Element-wise multiplication with x0 projection
        x0_proj = tf.matmul(x0, self.V)
        interaction = tf.multiply(x_proj, x0_proj)
        
        # Project back to original dimension and add bias
        out = tf.matmul(interaction, tf.transpose(self.U)) + self.bias
        
        # Add residual connection
        return x0 + out
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(CrossNetworkLayer, self).get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "kernel_initializer": self.kernel_initializer
        })
        return config

# Multi-head feature attention layer for capturing correlations between features
class MultiHeadFeatureAttention(Layer):
    def __init__(self, num_heads=4, key_dim=32, dropout=0.1, **kwargs):
        super(MultiHeadFeatureAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        
        # Layer normalization for stability
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(self.input_dim * 2, activation='swish'),
            Dropout(self.dropout_rate),
            Dense(self.input_dim)
        ])
        
        super(MultiHeadFeatureAttention, self).build(input_shape)
        
    def call(self, inputs, training=None, **kwargs):
        # Reshape to sequence format
        x = tf.expand_dims(inputs, axis=1)
        
        # Transpose to create feature-wise sequence
        feature_seq = tf.transpose(x, [0, 2, 1])
        
        # Self-attention on features
        attended_features = self.attention(
            feature_seq, feature_seq, feature_seq,
            training=training
        )
        
        # Add & normalize
        x1 = self.layer_norm1(feature_seq + attended_features)
        
        # Feed-forward & residual
        ffn_output = self.ffn(x1, training=training)
        x2 = self.layer_norm2(x1 + ffn_output)
        
        # Transpose back and remove sequence dimension
        return tf.squeeze(tf.transpose(x2, [0, 2, 1]), axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(MultiHeadFeatureAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate
        })
        return config

# Enhanced Feature Interaction Layer combining various methods
class EnhancedFeatureInteractionLayer(Layer):
    def __init__(self, hidden_units=128, dropout_rate=0.2, use_batch_norm=True, **kwargs):
        super(EnhancedFeatureInteractionLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # First-order feature transformation
        self.first_order = Dense(self.hidden_units, activation=None)
        
        # Second-order feature crossing
        self.cross_network = CrossNetworkLayer(projection_dim=self.hidden_units // 2)
        
        # Self-attention mechanism
        self.attention = MultiHeadFeatureAttention(
            num_heads=4, 
            key_dim=max(16, self.hidden_units // 8),
            dropout=self.dropout_rate
        )
        
        # Feature compression
        self.compression = Dense(self.hidden_units, activation=None)
        
        # Normalization and regularization
        if self.use_batch_norm:
            self.batch_norm1 = BatchNormalization()
            self.batch_norm2 = BatchNormalization()
        self.dropout = Dropout(self.dropout_rate)
        
        # Output integration
        self.integration = Dense(self.hidden_units, activation=None)
        
        super(EnhancedFeatureInteractionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None, **kwargs):
        # First-order pathway
        first_order = self.first_order(inputs)
        if self.use_batch_norm:
            first_order = self.batch_norm1(first_order, training=training)
        first_order = tf.keras.activations.swish(first_order)
        
        # Second-order pathway with cross network
        crossed_features = self.cross_network(inputs)
        compressed = self.compression(crossed_features)
        if self.use_batch_norm:
            compressed = self.batch_norm2(compressed, training=training)
        crossed_activated = tf.keras.activations.swish(compressed)
        
        # Attention pathway
        attention_features = self.attention(inputs, training=training)
        
        # Combine pathways
        combined = Concatenate()([
            first_order, 
            crossed_activated,
            attention_features
        ])
        
        # Final integration
        output = self.integration(combined)
        output = tf.keras.activations.swish(output)
        output = self.dropout(output, training=training)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_units)
    
    def get_config(self):
        config = super(EnhancedFeatureInteractionLayer, self).get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return config

# Custom loss function for ordinal regression
def ordinal_loss(y_true, y_pred):
    """
    Custom loss function for ordinal regression.
    y_true: Ground truth values (0.0, 0.25, 0.5, 0.75, 1.0)
    y_pred: Predicted cumulative probabilities for each threshold
    """
    # Convert y_true to class indices (0, 1, 2, 3, 4)
    y_true_class = tf.cast(y_true * 4, tf.int32)
    
    # One-hot encode with num_classes = 5
    y_true_onehot = tf.one_hot(y_true_class, 5)
    
    # Convert to binary labels for each threshold
    # For class k, thresholds 0 to k-1 should be crossed (1), and k to end should not be crossed (0)
    # Create a matrix of shape [batch_size, num_thresholds]
    threshold_labels = tf.cumsum(y_true_onehot[:, :0:-1], axis=1, reverse=True)
    
    # Binary cross-entropy for each threshold
    bce = tf.keras.losses.binary_crossentropy(threshold_labels, y_pred)
    
    # Mean over thresholds
    return tf.reduce_mean(bce, axis=-1)

# Custom metric to measure ordinal MAE
def ordinal_mae(y_true, y_pred):
    """
    Custom MAE metric for ordinal regression.
    Converts predicted probabilities back to ordinal value.
    """
    # Add a column of ones (last threshold is always crossed)
    pred_probs = tf.concat([y_pred, tf.ones_like(y_pred[:, :1])], axis=1)
    
    # Calculate class probabilities from cumulative probabilities
    class_probs = pred_probs[:, :-1] - pred_probs[:, 1:]
    
    # Get predicted class
    pred_classes = tf.cast(tf.argmax(class_probs, axis=1), tf.float32)
    
    # Convert to original scale
    predictions = pred_classes / 4.0
    
    # Calculate MAE
    return tf.reduce_mean(tf.abs(y_true - predictions))

# Build the enhanced ordinal model
def build_enhanced_ordinal_model(input_dim=310, num_classes=5, hidden_dim=256):
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Enhanced feature interaction block 1 - deeper representation
    x = EnhancedFeatureInteractionLayer(hidden_units=hidden_dim, dropout_rate=0.2)(inputs)
    
    # Enhanced feature interaction block 2 - deeper representation with skip connection
    x2 = EnhancedFeatureInteractionLayer(hidden_units=hidden_dim, dropout_rate=0.25)(x)
    x = Add()([x, x2])  # Skip connection
    
    # Layer for final feature extraction
    feature_layer = Dense(hidden_dim//2, activation='swish')(x)
    feature_layer = Dropout(0.3)(feature_layer)
    
    # Output logit (scalar)
    output_logit = Dense(1)(feature_layer)
    
    # Ordinal layer for ordinal regression
    outputs = OrdinalLayer(num_classes=num_classes)(output_logit)
    
    return Model(inputs=inputs, outputs=outputs)

# Create the model instance
model = build_enhanced_ordinal_model()