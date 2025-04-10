import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input


model = Sequential([
            # Input layer
            Input(shape=(42,), dtype=tf.uint8),
            
            # Convert to float32 for stability
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer - 5 classes for our target values
            Dense(5, activation='softmax')
        ])