import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.regularizers import l1_l2

model = Sequential([
    # Input layer
    Input(shape=(42,), dtype=tf.uint8),
    
    # Convert to float32 for stability
    tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
    
    # Hidden layers with different regularization approaches
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output layer
    Dense(5, activation='softmax')
], name="Regularized")
