import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_parquet('data/train.parquet')

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# Split the data into features and target
X = df.iloc[:, :-1]  # all columns except the last one
y = df.iloc[:, -1]   # the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Disable GPU temporarily to avoid memory issues
# Comment this out later if GPU works well
tf.config.set_visible_devices([], 'GPU')

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Safer data preparation
def prepare_data(X):
    # Drop columns that are all NaN
    X = X.dropna(axis=1, how='all')
    
    # Convert remaining to numeric, coercing errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with median or 0 if median fails
    for col in X.columns:
        if X[col].isna().all():
            # If all values are NaN, drop the column
            X = X.drop(columns=[col])
        elif X[col].isna().any():
            # Fill NaNs with median, or 0 if median can't be calculated
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    return X

try:
    # Prepare data
    X_train = prepare_data(X_train.copy())
    X_test = prepare_data(X_test.copy())
    
    # Make sure test has same columns as train
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    print(f"Original X_train shape: {X_train.shape}")
    
    # Feature selection - select top 100 most important features
    print("Reducing features to 100...")
    selector = SelectKBest(f_regression, k=100)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)
    
    print(f"Reduced X_train shape: {X_train_reduced.shape}")
    
    # Build a simpler model with fewer parameters
    model = Sequential([
        Dense(16, input_shape=(100,), activation='relu'),  # Smaller first layer
        Dense(8, activation='relu'),                       # Smaller second layer
        Dense(1, activation='linear')                      # Output layer
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train with smaller batch size and fewer epochs
    history = model.fit(
        X_train_reduced, y_train,
        epochs=10,                   # Reduced epochs
        batch_size=8,                # Smaller batch size
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    loss, mae = model.evaluate(X_test_reduced, y_test)
    print(f'Test Loss (MSE): {loss:.4f}')
    print(f'Test MAE: {mae:.4f}')
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()