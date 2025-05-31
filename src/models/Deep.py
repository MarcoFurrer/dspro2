import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Layer, Concatenate, Reshape, MultiHeadAttention, LayerNormalization

# Custom Feature Interaction Layer - Simplified version to avoid reshape errors
class FeatureInteractionLayer(Layer):
    def __init__(self, **kwargs):
        super(FeatureInteractionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Store the input shape for later use
        self.input_feature_dim = int(input_shape[1])
        self.interaction_dim = self.input_feature_dim * (self.input_feature_dim - 1) // 2
        super(FeatureInteractionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Use a simpler approach with element-wise multiplication for selected pairs
        batch_size = tf.shape(inputs)[0]
        feature_dim = self.input_feature_dim
        
        # Create a list to store all pairwise interactions
        interaction_features = []
        
        # Generate all unique pairs of features (upper triangular without diagonal)
        for i in range(feature_dim):
            for j in range(i+1, feature_dim):
                # Extract the features
                feature_i = tf.expand_dims(inputs[:, i], axis=1)
                feature_j = tf.expand_dims(inputs[:, j], axis=1)
                
                # Multiply the features
                interaction = feature_i * feature_j
                interaction_features.append(interaction)
        
        # Concatenate all interactions
        if interaction_features:
            interactions_tensor = tf.concat(interaction_features, axis=1)
            # Concatenate with original features
            return tf.concat([inputs, interactions_tensor], axis=1)
        else:
            return inputs
    
    def compute_output_shape(self, input_shape):
        feature_dim = int(input_shape[1])
        interaction_dim = feature_dim * (feature_dim - 1) // 2
        return (input_shape[0], feature_dim + interaction_dim)

# Custom Multi-Head Attention Block
def attention_block(inputs, num_heads=4, key_dim=8):
    # Get the input dimension explicitly
    input_dim = int(inputs.shape[-1])
    
    # Reshape inputs for attention layer - use fixed sequence length of 1
    x = Reshape((1, input_dim))(inputs)
    
    # Apply multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim
    )(x, x)
    
    # Reshape back to original dimensions with explicit shape
    attention_output = Reshape((input_dim,))(attention_output)
    
    # Add residual connection and normalize
    x = inputs + attention_output
    x = LayerNormalization()(x)
    
    return x

# Build a functional model instead of sequential for complex connections
inputs = Input(shape=(42,), dtype=tf.uint8)

# Convert to float32 for stability
x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)

# Apply feature interaction
x = FeatureInteractionLayer()(x)

# Initial dense layer
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Multi-head attention to focus on important feature combinations
x = attention_block(x, num_heads=4, key_dim=16)

# Continue with existing architecture
x = Dense(96, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# Another attention block
x = attention_block(x, num_heads=2, key_dim=24)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# Output layer
outputs = Dense(1, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)
