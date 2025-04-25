import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def create_deep_model(feature_count, n_categories=5):
    """Create a deeper model with more layers"""
    model = Sequential([
        Input(shape=(feature_count,), dtype=tf.uint8),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(n_categories, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_wide_model(feature_count, n_categories=5):
    """Create a wider model with more neurons per layer"""
    model = Sequential([
        Input(shape=(feature_count,), dtype=tf.uint8),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(n_categories, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_leaky_relu_model(feature_count, n_categories=5):
    """Create a model using LeakyReLU activation"""
    model = Sequential([
        Input(shape=(feature_count,), dtype=tf.uint8),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        
        Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(n_categories, activation='softmax')
    ])
    
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_regularized_model(feature_count, n_categories=5):
    """Create a highly regularized model with L1/L2 regularization"""
    model = Sequential([
        Input(shape=(feature_count,), dtype=tf.uint8),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        
        Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(n_categories, activation='softmax')
    ])
    
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
