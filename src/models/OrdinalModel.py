import tensorflow as tf
from keras.layers import (
    Layer, Dense, Input, Dropout, BatchNormalization, 
    Add, Activation, MultiHeadAttention, Reshape, 
    Concatenate, Multiply, GlobalAveragePooling1D, Conv1D
)
from keras.models import Model
import numpy as np

class OrdinalLayer(Layer):
    """
    Custom layer for ordinal regression - outputs cumulative probabilities
    for ordinal target values [0, 0.25, 0.5, 0.75, 1.0]
    """
    def __init__(self, num_classes=5, scaling_factor=15.0, **kwargs):
        self.num_classes = num_classes
        self.scaling_factor = scaling_factor
        self.supports_masking = True
        super(OrdinalLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create trainable thresholds for ordinal regression
        # Initialize with more gradual spacing to improve learning
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.num_classes - 1,),
            initializer=tf.keras.initializers.Constant(np.linspace(0.15, 0.85, self.num_classes - 1)),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99)  # Wider bounds
        )
        # Add a trainable scaling factor to adapt during training
        self.learned_scaling = self.add_weight(
            name='learned_scaling',
            shape=(1,),
            initializer=tf.keras.initializers.Constant([self.scaling_factor]),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 5.0, 30.0)  # Wider bounds for more flexibility
        )
        super(OrdinalLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        # Calculate probabilities of crossing each threshold
        # Use a learned scaling factor for adaptive scaling
        return tf.sigmoid(self.learned_scaling * (inputs - self.thresholds))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes - 1
    
    def get_config(self):
        config = super(OrdinalLayer, self).get_config()
        config.update({
            "num_classes": self.num_classes,
            "scaling_factor": self.scaling_factor
        })
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
        
        # Add residual connection with a scaling factor to control the information flow
        return x0 + 0.1 * out  # Scale down the interaction contribution
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(CrossNetworkLayer, self).get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "kernel_initializer": self.kernel_initializer
        })
        return config

# Simplified Multi-head feature attention layer
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
        
        # Simplified FFN to reduce overfitting
        self.ffn = tf.keras.Sequential([
            Dense(self.input_dim, activation='swish')
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
        x1 = self.layer_norm1(feature_seq + 0.2 * attended_features)  # Scale down attention
        
        # Remove sequence dimension
        return tf.squeeze(tf.transpose(x1, [0, 2, 1]), axis=1)
    
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

# Feature Interaction Layer with better initial learning
class FeatureInteractionLayer(Layer):
    def __init__(self, hidden_units=128, dropout_rate=0.2, use_batch_norm=True, **kwargs):
        super(FeatureInteractionLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # Direct feature transformation - ensures basic feature learning happens first
        self.direct_transform = Dense(self.hidden_units, activation='swish')
        
        # Simple pairwise feature interaction layer
        self.pairwise = Dense(self.hidden_units // 2, activation=None)
        
        # Normalization and regularization
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.dropout = Dropout(self.dropout_rate)
        
        super(FeatureInteractionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None, **kwargs):
        # Direct transformation - ensures basic feature learning
        direct = self.direct_transform(inputs)
        
        # Simple pairwise interactions
        pairwise = self.pairwise(inputs)
        
        # Combine pathways
        combined = Concatenate()([direct, pairwise])
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            combined = self.batch_norm(combined, training=training)
        
        # Apply activation and dropout
        output = tf.keras.activations.swish(combined)
        output = self.dropout(output, training=training)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_units + self.hidden_units // 2)
    
    def get_config(self):
        config = super(FeatureInteractionLayer, self).get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return config

# Enhanced Feature Interaction Layer combining various methods - for later stages
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
        
        # Combine pathways with weighted contribution to prevent initial training issues
        # Focus more on first-order features initially to establish a good baseline
        combined = Concatenate()([
            first_order, 
            crossed_activated * 0.7,  # Reduce contribution of complex features
            attention_features * 0.5  # Reduce contribution of attention features
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

# Improved custom loss function for ordinal regression
def ordinal_loss(y_true, y_pred):
    """
    Custom loss function for ordinal regression with focal loss component
    to address class imbalance and focus on difficult examples.
    """
    # Convert y_true to class indices (0, 1, 2, 3, 4)
    y_true_class = tf.cast(y_true * 4, tf.int32)
    
    # One-hot encode with num_classes = 5
    y_true_onehot = tf.one_hot(y_true_class, 5)
    
    # Convert to binary labels for each threshold
    threshold_labels = tf.cumsum(y_true_onehot[:, :0:-1], axis=1, reverse=True)
    
    # Binary cross-entropy for each threshold
    bce = tf.keras.losses.binary_crossentropy(threshold_labels, y_pred)
    
    # Add a focal loss component (gamma=2) to focus on hard examples
    p_t = threshold_labels * y_pred + (1 - threshold_labels) * (1 - y_pred)
    focal_weight = tf.pow(1.0 - p_t, 2.0)  # Square of the complement of p_t
    
    # Apply focal weighting to BCE
    focal_bce = focal_weight * bce
    
    # Add a small L2 regularization term to prevent overfitting on distribution
    l2_reg = 0.001 * tf.reduce_mean(tf.square(y_pred - 0.5))
    
    # Return weighted loss with regularization
    return tf.reduce_mean(focal_bce, axis=-1) + l2_reg

# Custom metric to measure ordinal MAE
def ordinal_mae(y_true, y_pred):
    """
    Custom MAE metric for ordinal regression.
    Converts predicted probabilities back to ordinal value.
    """
    # Add a column of ones (last threshold is always crossed)
    pred_probs = tf.concat([y_pred, tf.ones_like(y_pred[:, :1])], axis=1)
    
    # Calculate class probabilities from cumulative probabilities
    class_probs = tf.concat([
        pred_probs[:, :1],  # First class prob = first threshold prob
        pred_probs[:, :-1] - pred_probs[:, 1:],  # Middle classes
        1.0 - pred_probs[:, -1:],  # Last class prob = 1 - last threshold prob
    ], axis=1)
    
    # Get predicted class - use expected value for more stable training
    class_indices = tf.cast(tf.range(5), dtype=tf.float32)
    expected_class = tf.reduce_sum(class_probs * class_indices, axis=1)
    
    # Convert to original scale
    predictions = expected_class / 4.0
    
    # Calculate MAE
    return tf.reduce_mean(tf.abs(y_true - predictions))

# Build a progressive model that helps learn the basics first
def build_progressive_ordinal_model(input_dim=310, num_classes=5, hidden_dim=256):
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Stage 1: Basic feature learning with simpler architecture
    basic_features = FeatureInteractionLayer(hidden_units=hidden_dim//2, dropout_rate=0.1)(inputs)
    
    # Stage 2: Intermediate feature processing with moderate complexity
    x = Dense(hidden_dim, activation='swish')(basic_features)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Stage 3: Advanced feature learning with controlled complexity
    advanced_features = EnhancedFeatureInteractionLayer(hidden_units=hidden_dim, dropout_rate=0.25)(x)
    
    # Stage 4: Final feature refinement with skip connection for better gradient flow
    x = Dense(hidden_dim//2, activation='swish')(advanced_features)
    x = Dropout(0.3)(x)
    x = Add()([x, Dense(hidden_dim//2, activation='swish')(basic_features)])  # Skip connection to basic features
    
    # Output logit (scalar)
    output_logit = Dense(1)(x)
    
    # Ordinal layer for ordinal regression
    outputs = OrdinalLayer(num_classes=num_classes)(output_logit)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model instance
model = build_progressive_ordinal_model()

# Optional feature attention layer - separate implementation
class SelfAttentionLayer(Layer):
    """
    Self-attention layer for feature importance learning
    """
    def __init__(self, attention_dim=64, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.query_dense = Dense(self.attention_dim)
        self.key_dense = Dense(self.attention_dim)
        self.value_dense = Dense(input_shape[-1])
        
        super(SelfAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        # Compute query, key, value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Reshape for attention
        query_r = tf.expand_dims(query, axis=1)
        key_r = tf.expand_dims(key, axis=2)
        
        # Compute attention scores
        attention_scores = tf.matmul(query_r, key_r)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.attention_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention weights
        attended_values = attention_weights * value
        
        # Residual connection
        output = inputs + 0.3 * attended_values  # Scale down to not overpower direct features
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({"attention_dim": self.attention_dim})
        return config