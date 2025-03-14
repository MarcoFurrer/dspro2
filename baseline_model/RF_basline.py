import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load Numerai Data (Replace with actual Numerai dataset)
df = pd.read_csv("numerai_dataset.csv")  # Ensure the dataset is downloaded

# Step 2: Feature Engineering - Creating Lag Features
df["lag_1"] = df["target"].shift(1)  # Previous day's target
df["lag_2"] = df["target"].shift(2)  # Two days ago
df["rolling_mean_5"] = df["target"].rolling(window=5).mean()  # 5-day moving average
df["rolling_std_5"] = df["target"].rolling(window=5).std()  # Rolling std deviation

# Drop NaN values from shifting
df.dropna(inplace=True)

# Step 3: Prepare Data for Training
feature_cols = ["lag_1", "lag_2", "rolling_mean_5", "rolling_std_5"]  # Select features
X = df[feature_cols]  # Feature matrix
y = df["target"]  # Target variable

# Train-test split (IMPORTANT: No shuffling for time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
model.fit(X_train, y_train)

# Step 5: Make Predictions
preds = model.predict(X_test)

# Step 6: Evaluate Model Performance
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse:.4f}")