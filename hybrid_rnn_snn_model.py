"""
HYBRID RNN-SNN ELECTRICITY LOAD FORECASTING
=====================================================

"""


print("STEP 1: Installing and setting up libraries...")
!pip install snntorch -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import json
warnings.filterwarnings('ignore')

print("✓ Libraries loaded successfully!")


def load_and_preprocess_data_corrected():
    """
    Corrected data loading and preprocessing pipeline.
    - Removes invalid data points.
    - Uses MinMaxScaler for robust normalization.
    - Implements a chronological train/test split (no shuffling).
    """
    print("\nSTEP 2: Loading and preprocessing data with corrections...")

    # Load raw data
    try:
        df_raw = pd.read_csv('hourly_load_2016.csv', header=None)
    except FileNotFoundError:
        print("ERROR: 'hourly_load_2016.csv' not found. Please upload the file.")
        return None, None, None

    # Vectorized extraction of hourly data (faster and cleaner)
    hourly_loads = df_raw[df_raw.index % 24 != 0].iloc[:, 1].values

    # Clean data: remove NaNs and non-positive values
    hourly_loads = hourly_loads[~np.isnan(hourly_loads) & (hourly_loads > 0)]
    hourly_data = hourly_loads.reshape(-1, 1)

    # Normalize data using MinMaxScaler - this is standard practice and avoids MAPE issues
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hourly_data).flatten()

    print(f"✓ Loaded and cleaned {len(scaled_data)} hourly data points.")

    # Create sequences from the scaled data
    sequence_length = 24
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X, y = np.array(X), np.array(y)

    # Chronological train/test split - ESSENTIAL for time series
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"✓ Correct chronological split applied:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples: {X_test.shape[0]}")

    return (X_train, y_train), (X_test, y_test), scaler


def build_stable_rnn_model(input_shape):
    """
    Builds a robust and proven stacked LSTM model. This replaces the complex
    and unstable dual-stage attention model.
    """
    print("\nSTEP 3: Building a stable and high-performance RNN (LSTM) model...")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.summary()
    return model

def train_rnn_model(model, X_train, y_train):
    """Trains the RNN model with best practices like early stopping."""
    print("✓ Training the RNN model...")

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    print("✓ RNN training complete!")
    return history


class OptimizedSNN(nn.Module):
    """An optimized and stable SNN architecture."""
    def __init__(self, input_size, hidden_size=128, num_steps=20):
        super().__init__()
        self.num_steps = num_steps

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.fc2 = nn.Linear(hidden_size, 1) # Simplified output layer

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        output_potentials = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            output = self.fc2(spk1)
            output_potentials.append(output)

        return torch.stack(output_potentials).mean(0)

def train_snn_model(X_train, y_train, X_test):
    """Trains the optimized SNN with a robust training loop."""
    print("\nSTEP 4: Training the optimized SNN model...")

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = OptimizedSNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Adam is generally more stable

    num_epochs = 50 # Reduced epochs as it should converge faster
    batch_size = 128

    for epoch in range(num_epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            model.train()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"  SNN Epoch {epoch}, Loss: {loss.item():.6f}")

    print("✓ SNN training complete!")

    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)

    return predictions.numpy().flatten()


def create_hybrid_predictions(rnn_pred, snn_pred, alpha=0.5):
    """Combines predictions with a simple, effective weighted average."""
    print("\nSTEP 5: Creating the Hybrid Model...")

    # Ensure prediction arrays are the same length
    min_len = min(len(rnn_pred), len(snn_pred))

    hybrid_pred = alpha * rnn_pred[:min_len] + (1 - alpha) * snn_pred[:min_len]

    print(f"✓ Hybrid predictions created with RNN weight {alpha} and SNN weight {1-alpha}.")
    return rnn_pred[:min_len], snn_pred[:min_len], hybrid_pred


def evaluate_and_visualize(y_test, rnn_pred, snn_pred, hybrid_pred, scaler):
    """Performs final evaluation and creates professional plots."""
    print("\nSTEP 6: Evaluating models and creating visualizations...")

    # Inverse transform predictions and actuals to their original scale
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rnn_pred_orig = scaler.inverse_transform(rnn_pred.reshape(-1, 1)).flatten()
    snn_pred_orig = scaler.inverse_transform(snn_pred.reshape(-1, 1)).flatten()
    hybrid_pred_orig = scaler.inverse_transform(hybrid_pred.reshape(-1, 1)).flatten()

    def calculate_metrics(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"\n{model_name} Performance:")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - MAPE: {mape:.2f}%")
        return {'MAE': mae, 'MAPE': mape}

    rnn_metrics = calculate_metrics(y_test_orig, rnn_pred_orig, "RNN (LSTM)")
    snn_metrics = calculate_metrics(y_test_orig, snn_pred_orig, "Real SNN")
    hybrid_metrics = calculate_metrics(y_test_orig, hybrid_pred_orig, "Hybrid RNN-SNN")

    # Visualization
    plt.figure(figsize=(18, 10))
    n_display = 400

    plt.plot(y_test_orig[:n_display], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(rnn_pred_orig[:n_display], 'b--', label=f"RNN (MAPE: {rnn_metrics['MAPE']:.2f}%)", alpha=0.7)
    plt.plot(snn_pred_orig[:n_display], 'r:', label=f"SNN (MAPE: {snn_metrics['MAPE']:.2f}%)", alpha=0.7)
    plt.plot(hybrid_pred_orig[:n_display], 'g-', label=f"Hybrid (MAPE: {hybrid_metrics['MAPE']:.2f}%)", linewidth=2.5)

    plt.title("Optimized Hybrid RNN-SNN Load Forecasting", fontsize=16, fontweight='bold')
    plt.xlabel("Hour in Test Set")
    plt.ylabel("Electricity Load (Original Scale)")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('final_submission_results.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Corrected data pipeline
    (X_train, y_train), (X_test, y_test), scaler = load_and_preprocess_data_corrected()

    if scaler is not None:
        # Train RNN Model
        rnn_model = build_stable_rnn_model((X_train.shape[1], 1))
        rnn_history = train_rnn_model(rnn_model, X_train, y_train)

        # Train SNN Model
        snn_predictions = train_snn_model(X_train, y_train, X_test)

        # Get RNN predictions
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        rnn_predictions = rnn_model.predict(X_test_reshaped).flatten()

        # Create and evaluate hybrid
        rnn_pred_final, snn_pred_final, hybrid_pred_final = create_hybrid_predictions(rnn_predictions, snn_predictions)

        # Adjust y_test length to match predictions
        y_test_final = y_test[:len(hybrid_pred_final)]

        # Final evaluation and visualization
        evaluate_and_visualize(y_test_final, rnn_pred_final, snn_pred_final, hybrid_pred_final, scaler)

        print("\n" + "="*60)
        print("PROJECT EXECUTION COMPLETE.")
        print("="*60)
