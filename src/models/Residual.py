import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Add

# Input layer
inputs = Input(shape=(30,), dtype=tf.uint8)
x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)

# First block
x1 = Dense(128, activation='relu')(x)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.3)(x1)

# Second block with residual connection
x2 = Dense(128, activation='relu')(x1)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.3)(x2)
x2 = Add()([x1, x2])  # Residual connection

# Third block with residual connection
x3 = Dense(128, activation='relu')(x2)
x3 = BatchNormalization()(x3)
x3 = Dropout(0.3)(x3)
x3 = Add()([x2, x3])  # Residual connection

# Output layer
outputs = Dense(5, activation='softmax')(x3)

model = Model(inputs=inputs, outputs=outputs)
