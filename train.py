import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- 1. Load Data from JSON ---
def load_enrollment_data():
    try:
        with open('dataset.json', 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data['enrollment_data'])
            print("Successfully loaded data from dataset.json")
            return df
    except FileNotFoundError:
        print("Error: dataset.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in dataset.json")
        return None

# Load the data
df = load_enrollment_data()
if df is None:
    raise SystemExit("Failed to load the dataset. Please ensure dataset.json exists and is valid.")

print("Sample Data Head:")
print(df.head())
print("\nData Info:")
df.info()

# --- 2. Data Preprocessing ---

# Define features (X) and target (y)
target_col = 'Actual_Enrollment'

# Identify categorical and numerical features (excluding target and time identifiers)
categorical_features = ['Program']
# Exclude Year and Quarter initially, will be used for sequencing
numerical_features = [col for col in df.columns if df[col].dtype != 'object' and col not in [target_col, 'Year', 'Quarter']]

print(f"\nNumerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}")
print(f"Target Feature: {target_col}")

# Create preprocessing pipelines for numerical and categorical features
# Numerical features: Scale to range [0, 1]
# Categorical features: One-hot encode
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like Year, Quarter, Target) for now
)

# Separate target variable scaler (important for inverse transform later)
target_scaler = MinMaxScaler()
df[[target_col]] = target_scaler.fit_transform(df[[target_col]])

# Apply preprocessing to features (excluding target)
# Fit the preprocessor on the entire dataset to learn categories and scaling ranges
# Note: In a real scenario, fit ONLY on training data to avoid data leakage
# For simplicity here, we fit on all data before sequencing and splitting.
# A more robust approach involves splitting first, then fitting preprocessor on train data.
features_processed = preprocessor.fit_transform(df.drop(columns=[target_col]))

# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Combine processed features with Year, Quarter, and the scaled Target
df_processed = pd.DataFrame(features_processed, columns=all_feature_names + ['Year', 'Quarter']) # Adjust column names if needed
df_processed[target_col] = df[target_col] # Add the scaled target back

print("\nProcessed Data Head:")
print(df_processed.head())

# --- 3. Create Sequences for LSTM ---
# LSTM requires input data in the form of sequences [samples, time_steps, features]

def create_sequences(data, target_col, sequence_length, feature_cols, group_col='Program'):
    X, y = [], []
    # Group data by program to create sequences within each program's timeline
    grouped = data.groupby(group_col)
    for _, group_df in grouped:
        features = group_df[feature_cols].values
        target = group_df[target_col].values
        for i in range(len(group_df) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(target[i + sequence_length]) # Predict the next step's enrollment
    return np.array(X), np.array(y)

sequence_length = 4 # Use past 4 quarters (1 year) to predict the next quarter
feature_columns_for_lstm = all_feature_names # Use all processed features

# Create sequences using the processed data (excluding Year/Quarter now)
X_seq, y_seq = create_sequences(df_processed, target_col, sequence_length, feature_columns_for_lstm, group_col=None) # Grouping was handled conceptually before

print(f"\nShape of sequence features (X_seq): {X_seq.shape}") # (num_sequences, sequence_length, num_features)
print(f"Shape of sequence target (y_seq): {y_seq.shape}")     # (num_sequences,)

# --- 4. Split Data (Train/Test) ---
# Important: For time series, split chronologically if possible.
# Since we mixed programs during sequencing, a standard split is okay here,
# but a chronological split *within each program* before sequencing is ideal.
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True) # Shuffle=True is okay post-sequencing

print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- 5. Build LSTM Model ---
n_features = X_train.shape[2] # Number of features per time step

model = Sequential([
    LSTM(units=100, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True), # Return sequences for stacking LSTMs
    Dropout(0.2),
    LSTM(units=50, activation='relu', return_sequences=False), # Only the last LSTM layer returns a single output
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1) # Output layer: 1 neuron for predicting 'Actual_Enrollment' (linear activation is default)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # Mean Absolute Error is often interpretable

model.summary()

# --- 6. Train the Model ---
# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100, # Adjust number of epochs as needed
    batch_size=32, # Adjust batch size based on memory/performance
    validation_split=0.2, # Use part of the training data for validation during training
    callbacks=[early_stopping],
    verbose=1 # Set to 0 for less output, 1 for progress bar
)

# --- 7. Evaluate the Model ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()


# --- 8. Make Predictions & Inverse Transform ---
# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform the scaled predictions and actual values to get original enrollment numbers
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Compare some predictions with actual values
comparison_df = pd.DataFrame({'Actual Enrollment': y_test_actual.flatten(), 'Predicted Enrollment': y_pred.flatten()})
print("\nSample Predictions vs Actual Values:")
print(comparison_df.head(10))

# Optional: Plot predictions vs actual
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Actual Enrollment', marker='.', linestyle='None', alpha=0.7)
plt.plot(y_pred, label='Predicted Enrollment', marker='x', linestyle='None', alpha=0.7)
plt.title('Test Set: Actual vs Predicted Enrollment')
plt.xlabel('Sample Index (Test Set)')
plt.ylabel('Number of Students')
plt.legend()
plt.grid(True)
plt.show()

# --- 9. Predict on New Data (Example) ---
# To predict for a future quarter, you need the sequence of the *last* `sequence_length`
# quarters' data for the specific program, preprocessed in the *exact same way*
# as the training data.

# Example: Get the last sequence from the original processed dataframe (demonstration only)
# In reality, you'd construct this from new incoming data.
last_sequence_processed = df_processed[feature_columns_for_lstm].values[-sequence_length:]

# Reshape for LSTM input: (1 sample, sequence_length, n_features)
last_sequence_reshaped = np.expand_dims(last_sequence_processed, axis=0)

# Predict the scaled value
next_enrollment_scaled = model.predict(last_sequence_reshaped)

# Inverse transform to get the actual predicted enrollment number
next_enrollment_predicted = target_scaler.inverse_transform(next_enrollment_scaled)

print(f"\nPredicted Enrollment for the next time step (based on last sequence): {next_enrollment_predicted[0][0]:.0f}")

# --- Optional: Save the Model ---
# model.save('enrollment_lstm_model.h5')
# print("\nModel saved to enrollment_lstm_model.h5")

# To load later:
# from tensorflow.keras.models import load_model
# loaded_model = load_model('enrollment_lstm_model.h5')
# print("\nModel loaded.")
