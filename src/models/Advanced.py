import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input


model = Sequential([
            # Input layer - use float32 for continuous features
            Input(shape=(42,), dtype=tf.float32),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer - 1 neuron for continuous regression [0,1]
            Dense(1, activation='sigmoid')
        ], name="Advanced")