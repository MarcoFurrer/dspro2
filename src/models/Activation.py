import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input

model = Sequential([
    # Input layer
    Input(shape=(42,), dtype=tf.uint8),
    
    # Convert to float32 for stability
    tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
    
    # Hidden layers with different activations
    Dense(128, activation='elu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='selu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='tanh'),
    BatchNormalization(),
    Dropout(0.1),
    
    # Output layer
    Dense(5, activation='softmax')
], name="Activation")
