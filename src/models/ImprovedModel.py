import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Dense, BatchNormalization, Dropout, Input, 
    Concatenate, LayerNormalization, Multiply, 
    LeakyReLU, Add, GlobalAveragePooling1D, Reshape
)
from keras.regularizers import l1_l2

class FeatureExpander(tf.keras.layers.Layer):
    """Custom layer to create polynomial and interaction features"""
    
    def __init__(self, polynomial_degree=2, **kwargs):
        super(FeatureExpander, self).__init__(**kwargs)
        self.polynomial_degree = polynomial_degree
        
    def build(self, input_shape):
        self.input_dim = int(input_shape[1])
        super(FeatureExpander, self).build(input_shape)
        
    def call(self, inputs):
        # Original features
        expanded_features = [inputs]
        
        # Polynomial features (higher degree for capturing non-linear relationships)
        for degree in range(2, self.polynomial_degree + 1):
            poly_features = tf.pow(inputs, degree)
            expanded_features.append(poly_features)
        
        # Log-transform features (adding small epsilon to avoid log(0))
        log_features = tf.math.log(tf.abs(inputs) + 1e-10)
        expanded_features.append(log_features)
        
        # Root features (square root and cube root)
        sqrt_features = tf.math.sqrt(tf.abs(inputs) + 1e-10)
        expanded_features.append(sqrt_features)
        
        cbrt_features = tf.pow(tf.abs(inputs) + 1e-10, 1/3)
        expanded_features.append(cbrt_features)
        
        # Selected pairwise interactions (to avoid dimension explosion)
        feature_count = self.input_dim
        max_interactions = min(feature_count, 25)  # Increased from 20 to 25
        
        for i in range(max_interactions):
            for j in range(i+1, max_interactions):
                interaction = tf.expand_dims(inputs[:, i], axis=1) * tf.expand_dims(inputs[:, j], axis=1)
                expanded_features.append(interaction)
                
        # Concatenate all expanded features
        return tf.concat(expanded_features, axis=1)
        
    def compute_output_shape(self, input_shape):
        feature_count = int(input_shape[1])
        max_interactions = min(feature_count, 25)
        interaction_count = (max_interactions * (max_interactions - 1)) // 2
        polynomial_count = self.polynomial_degree - 1  # Not counting original features
        
        # Original + polynomial + log + sqrt + cbrt + interactions
        return (input_shape[0], feature_count * (3 + polynomial_count) + interaction_count)

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with skip connections"""
    
    def __init__(self, units, dropout_rate=0.3, l1_reg=1e-5, l2_reg=1e-4, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Regularizer for all Dense layers
        regularizer = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        
        # First branch
        self.dense1 = Dense(self.units, activation=None, kernel_regularizer=regularizer)
        self.bn1 = BatchNormalization()
        self.leaky1 = LeakyReLU(alpha=0.1)
        self.dropout1 = Dropout(self.dropout_rate)
        
        # Second branch
        self.dense2 = Dense(self.units, activation=None, kernel_regularizer=regularizer)
        self.bn2 = BatchNormalization()
        self.leaky2 = LeakyReLU(alpha=0.1)
        self.dropout2 = Dropout(self.dropout_rate)
        
        # Skip connection (if dimensions don't match)
        self.skip_connection = None
        if input_dim != self.units:
            self.skip_connection = Dense(self.units, activation=None, kernel_regularizer=regularizer)
            
        super(ResidualBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        # First branch
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.leaky1(x)
        x = self.dropout1(x, training=training)
        
        # Second branch
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.skip_connection is not None:
            skip = self.skip_connection(inputs)
        else:
            skip = inputs
            
        # Add skip connection
        x = Add()([x, skip])
        x = self.leaky2(x)
        x = self.dropout2(x, training=training)
        
        return x

# Create the improved model
def build_model():
    # Input layer
    inputs = Input(shape=(42,), dtype=tf.uint8)
    
    # Convert to float32 and normalize
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)
    x = LayerNormalization()(x)
    
    # Expanded feature engineering
    x = FeatureExpander(polynomial_degree=3)(x)
    
    # Deeper network with residual connections
    x = ResidualBlock(384, dropout_rate=0.5, l1_reg=1e-6, l2_reg=1e-5)(x)
    x = ResidualBlock(256, dropout_rate=0.4, l1_reg=1e-6, l2_reg=1e-5)(x)
    x = ResidualBlock(128, dropout_rate=0.3, l1_reg=1e-6, l2_reg=1e-5)(x)
    x = ResidualBlock(64, dropout_rate=0.2, l1_reg=1e-6, l2_reg=1e-5)(x)
    
    # Output layer with regularization
    outputs = Dense(
        1, 
        activation='softmax',
        kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Initialize the model
model = build_model()