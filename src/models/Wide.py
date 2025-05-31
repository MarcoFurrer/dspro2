import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input

model = Sequential([
    # Input layer
    Input(shape=(42,), dtype=tf.uint8),
    
    # Convert to float32 for stability
    tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
    
    # Hidden layers - wider architecture
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(192, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output layer
    Dense(1, activation='softmax')
], name="Wide")
