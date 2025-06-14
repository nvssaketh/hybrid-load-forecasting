!pip install snntorch torch torchvision -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')
import json

print("="*60)
print("TRUE HYBRID CNN-SNN LOAD FORECASTING MODEL")
print("="*60)
print(" All libraries loaded successfully!")


def load_and_preprocess_data():
    """
    Load and preprocess data exactly as in your original load_forecasting.py
    """
    print("\n" + "="*50)
    print("DATA LOADING AND PREPROCESSING")
    print("="*50)

    # Load raw data (same as your original code)
    df_raw = pd.read_csv('hourly_load_2016.csv', header=None)
    df_raw_array = df_raw.values

    # Extract hourly load (same logic as your code)
    list_hourly_load = [df_raw_array[i,1]/100000 for i in range(0, len(df_raw)) if i % 24 != 0]

    print(f" Loaded {len(list_hourly_load)} hourly data points")
    print(f"  - Load range: {min(list_hourly_load):.2f} to {max(list_hourly_load):.2f} (×1e5)")

    return np.array(list_hourly_load)

def create_sequences(data, sequence_length=23):
    """
    Create sequences exactly as in your original load_forecasting.py
    """
    print(f"\n Creating sequences with {sequence_length} hour lookback")

    # Convert series to matrix (same as your convertSeriesToMatrix function)
    matrix = []
    for i in range(len(data) - sequence_length + 1):
        matrix.append(data[i:i + sequence_length])

    matrix = np.array(matrix)

    # Shift data by mean (same as your original code)
    shifted_value = matrix.mean()
    matrix -= shifted_value

    # Split dataset: 90% training, 10% testing (same as your original)
    train_row = int(round(0.9 * matrix.shape[0]))
    train_set = matrix[:train_row, :]

    # Shuffle training set (same as original)
    np.random.shuffle(train_set)

    # Create train/test splits
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = matrix[train_row:, :-1]
    y_test = matrix[train_row:, -1]

    print(f" Data split completed:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples: {X_test.shape[0]}")
    print(f"  - Shifted value: {shifted_value:.4f}")

    return (X_train, y_train), (X_test, y_test), shifted_value

def build_cnn_model(input_shape):
    """
    Enhanced CNN model based on your original LSTM approach
    """
    print("\n" + "="*50)
    print("CNN MODEL CONSTRUCTION")
    print("="*50)

    model = Sequential([
        # Convolutional layers for pattern detection
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(filters=100, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Conv1D(filters=50, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Dense layers (similar to your original LSTM structure)
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')  # Same as your original output
    ])

    # Compile with same settings as your original
    model.compile(loss="mse", optimizer="rmsprop")

    print(" CNN Model Architecture:")
    model.summary()

    return model

def train_cnn_model(model, X_train, y_train):
    """
    Train CNN with same parameters as your original LSTM
    """
    print("\n Training CNN model...")

    # Reshape for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Train with same parameters as your original
    history = model.fit(
        X_train_cnn, y_train,
        batch_size=512,  # Same as your original
        epochs=50,       # Same as your original
        validation_split=0.05,  # Same as your original
        verbose=1
    )

    print(" CNN training completed!")
    return history


class RealSNN(nn.Module):
    """
    Real Spiking Neural Network using snnTorch
    Implements genuine LIF neurons with membrane dynamics
    """
    def __init__(self, input_size, hidden_size, output_size, num_steps=25):
        super(RealSNN, self).__init__()

        self.num_steps = num_steps
        self.hidden_size = hidden_size

        # Network layers with LIF neurons
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.fc3 = nn.Linear(hidden_size//2, output_size)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x):
        """
        Forward pass through spiking neurons
        """
        batch_size = x.size(0)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record membrane potentials for output
        mem3_rec = []

        # Simulate spiking dynamics over time steps
        for step in range(self.num_steps):
            # Layer 1: Input -> Hidden
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2: Hidden -> Hidden
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3: Hidden -> Output
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            mem3_rec.append(mem3)

        # Use final membrane potential for prediction
        return torch.stack(mem3_rec, dim=0)

def train_real_snn(X_train, y_train, X_test, y_test):
    """
    Train the real SNN model
    """
    print("\n" + "="*50)
    print("REAL SNN MODEL TRAINING")
    print("="*50)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)

    # Create SNN model
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 1
    num_steps = 25

    snn_model = RealSNN(input_size, hidden_size, output_size, num_steps)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(snn_model.parameters(), lr=0.001)  # Same as CNN

    print(f" Real SNN Model created:")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Time steps: {num_steps}")
    print(f"  - Using genuine LIF neurons")

    # Training loop
    num_epochs = 50  # Same as CNN
    batch_size = 512  # Same as CNN

    print(f"\n Training SNN for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        snn_model.train()
        total_loss = 0

        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()

            # Forward pass
            mem_rec = snn_model(batch_X)
            output = mem_rec[-1]  # Use final membrane potential

            # Calculate loss
            loss = criterion(output, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            print(f"  Epoch {epoch}, Loss: {avg_loss:.6f}")

    print(" Real SNN training completed!")

    # Get predictions
    snn_model.eval()
    with torch.no_grad():
        mem_rec = snn_model(X_test_tensor)
        snn_predictions = mem_rec[-1].numpy().flatten()

    return snn_model, snn_predictions

def create_true_hybrid_predictions(cnn_model, snn_predictions, X_test, alpha=0.6):
    """
    Create true hybrid predictions combining CNN and real SNN
    """
    print("\n" + "="*50)
    print("TRUE HYBRID CNN-SNN MODEL")
    print("="*50)

    # Get CNN predictions
    print(" Generating CNN predictions...")
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    cnn_predictions = cnn_model.predict(X_test_cnn, verbose=0).flatten()

    print(" Combining CNN and real SNN predictions...")

    # Ensure same length
    min_length = min(len(cnn_predictions), len(snn_predictions))
    cnn_pred = cnn_predictions[:min_length]
    snn_pred = snn_predictions[:min_length]

    # Create hybrid predictions
    hybrid_predictions = alpha * cnn_pred + (1 - alpha) * snn_pred

    print(f" True hybrid predictions created:")
    print(f"  - CNN weight (α): {alpha}")
    print(f"  - Real SNN weight (1-α): {1-alpha}")
    print(f"  - Prediction samples: {len(hybrid_predictions)}")
    print(f"  - Using genuine spiking neurons with membrane dynamics")

    return cnn_pred, snn_pred, hybrid_predictions


def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"\n{model_name} Performance:")
    print(f"  - MSE:  {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE:  {mae:.6f}")
    print(f"  - MAPE: {mape:.2f}%")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def main():
    # Load and prepare data
    hourly_data = load_and_preprocess_data()
    (X_train, y_train), (X_test, y_test), shifted_value = create_sequences(hourly_data)

    # Build and train CNN
    cnn_model = build_cnn_model((X_train.shape[1], 1))
    cnn_history = train_cnn_model(cnn_model, X_train, y_train)

    # Train the real SNN
    snn_model, snn_predictions = train_real_snn(X_train, y_train, X_test, y_test)

    # Generate true hybrid predictions
    cnn_pred, snn_pred, hybrid_pred = create_true_hybrid_predictions(cnn_model, snn_predictions, X_test)

    # Adjust y_test to match prediction length
    y_test_adjusted = y_test[:len(hybrid_pred)]

    # Evaluate all models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    cnn_metrics = evaluate_model(y_test_adjusted, cnn_pred, "CNN")
    snn_metrics = evaluate_model(y_test_adjusted, snn_pred, "Real SNN")
    hybrid_metrics = evaluate_model(y_test_adjusted, hybrid_pred, "True Hybrid CNN-SNN")

    # Create comprehensive visualization
    plt.figure(figsize=(20, 12))

    # Plot 1: Overall comparison (similar to your original output)
    plt.subplot(2, 3, 1)
    n_display = min(500, len(y_test_adjusted))
    hours = np.arange(n_display)

    # Add shifted value back for display (same as your original)
    plt.plot(hours, (y_test_adjusted[:n_display] + shifted_value), 'k-', linewidth=2.5,
             label='Ground Truth', alpha=0.8)
    plt.plot(hours, (cnn_pred[:n_display] + shifted_value), 'b-', linewidth=1.5,
             label='CNN', alpha=0.7)
    plt.plot(hours, (snn_pred[:n_display] + shifted_value), 'r-', linewidth=1.5,
             label='Real SNN', alpha=0.7)
    plt.plot(hours, (hybrid_pred[:n_display] + shifted_value), 'g-', linewidth=2,
             label='True Hybrid CNN-SNN', alpha=0.8)

    plt.title('True Hybrid CNN-SNN Load Forecasting', fontsize=16, fontweight='bold')
    plt.xlabel('Hour')
    plt.ylabel('Electricity Load (×1e5)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Peak detection focus
    plt.subplot(2, 3, 2)
    peak_start, peak_end = 100, 200
    if peak_end < len(y_test_adjusted):
        hours_peak = np.arange(peak_start, peak_end)
        plt.plot(hours_peak, (y_test_adjusted[peak_start:peak_end] + shifted_value),
                 'k-', linewidth=3, label='Ground Truth')
        plt.plot(hours_peak, (cnn_pred[peak_start:peak_end] + shifted_value),
                 'b--', linewidth=2, label='CNN')
        plt.plot(hours_peak, (snn_pred[peak_start:peak_end] + shifted_value),
                 'r:', linewidth=2, label='Real SNN')
        plt.plot(hours_peak, (hybrid_pred[peak_start:peak_end] + shifted_value),
                 'g-', linewidth=2, label='True Hybrid')

        plt.title('Peak Detection: Real SNN vs CNN', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Electricity Load (×1e5)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: Performance comparison
    plt.subplot(2, 3, 3)
    models = ['CNN', 'Real SNN', 'True Hybrid']
    mse_values = [cnn_metrics['MSE'], snn_metrics['MSE'], hybrid_metrics['MSE']]
    mae_values = [cnn_metrics['MAE'], snn_metrics['MAE'], hybrid_metrics['MAE']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = plt.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8, color='lightcoral')

    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: SNN membrane potential dynamics
    plt.subplot(2, 3, 4)
    # Show membrane potential evolution for a sample
    sample_idx = 0
    sample_input = torch.FloatTensor(X_test[sample_idx:sample_idx+1])

    with torch.no_grad():
        mem_rec = snn_model(sample_input)
        membrane_evolution = mem_rec[:, 0, 0].numpy()  # First neuron, first sample

    time_steps = np.arange(len(membrane_evolution))
    plt.plot(time_steps, membrane_evolution, 'r-', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Membrane Potential')
    plt.title('Real SNN Membrane Dynamics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 5: Training history comparison
    plt.subplot(2, 3, 5)
    plt.plot(cnn_history.history['loss'], label='CNN Training Loss', linewidth=2)
    plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Error distribution
    plt.subplot(2, 3, 6)
    hybrid_errors = y_test_adjusted - hybrid_pred
    plt.hist(hybrid_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('True Hybrid Model Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('true_hybrid_cnn_snn_results.png', dpi=300, bbox_inches='tight')
    plt.show()


    print("\n" + "="*60)
    print("FINAL INTERNSHIP PROJECT REPORT")
    print("="*60)

    report = f"""
    HYBRID CNN-SNN ELECTRICITY LOAD FORECASTING
================================================


TECHNICAL IMPLEMENTATION:
- CNN: Enhanced version of original LSTM approach with convolutional layers
- Real SNN: Implemented using snnTorch with genuine spiking neurons
    * Leaky Integrate-and-Fire (LIF) neuron model
    * Membrane potential dynamics over {snn_model.num_steps} time steps
    * Spike generation and temporal processing
    * Biologically realistic neural computation
- Hybrid: Weighted combination (60% CNN, 40% Real SNN)

DATA PROCESSING:
- Dataset: {len(hourly_data)} hourly electricity load points from 2016
- Preprocessing: Same methodology as original load_forecasting.py
- Sequence length: {X_train.shape[1]} hours for prediction
- Train/test split: 90%/10% (same as original)
- Data normalization: Mean-centered (shifted value: {shifted_value:.4f})

MODEL PERFORMANCE:
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│     Model   │     MSE     │    RMSE     │     MAE     │    MAPE     │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ CNN         │ {cnn_metrics['MSE']:.6f} │ {cnn_metrics['RMSE']:.6f} │ {cnn_metrics['MAE']:.6f} │ {cnn_metrics['MAPE']:>6.2f}%   │
│ Real SNN    │ {snn_metrics['MSE']:.6f} │ {snn_metrics['RMSE']:.6f} │ {snn_metrics['MAE']:.6f} │ {snn_metrics['MAPE']:>6.2f}%   │
│ True Hybrid │ {hybrid_metrics['MSE']:.6f} │ {hybrid_metrics['RMSE']:.6f} │ {hybrid_metrics['MAE']:.6f} │ {hybrid_metrics['MAPE']:>6.2f}%   │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘


"""

    print(report)

    # Save comprehensive results
    results_df = pd.DataFrame({
        'Actual': y_test_adjusted + shifted_value,
        'CNN_Prediction': cnn_pred + shifted_value,
        'Real_SNN_Prediction': snn_pred + shifted_value,
        'True_Hybrid_Prediction': hybrid_pred + shifted_value,
        'CNN_Error': y_test_adjusted - cnn_pred,
        'SNN_Error': y_test_adjusted - snn_pred,
        'Hybrid_Error': y_test_adjusted - hybrid_pred
    })

    results_df.to_csv('true_hybrid_cnn_snn_internship_results.csv', index=False)

    # Save metrics
    all_metrics = {
        'CNN': cnn_metrics,
        'Real_SNN': snn_metrics,
        'True_Hybrid_CNN_SNN': hybrid_metrics,
        'Data_Info': {
            'total_samples': len(hourly_data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'sequence_length': X_train.shape[1],
            'shifted_value': float(shifted_value)
        }
    }

    with open('internship_model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    with open('internship_project_report.txt', 'w') as f:
        f.write(report)

    print("\n Internship deliverables:")
    print("  - true_hybrid_cnn_snn_internship_results.csv")
    print("  - internship_model_metrics.json")
    print("  - internship_project_report.txt")
    print("  - true_hybrid_cnn_snn_results.png")

if __name__ == "__main__":
    main()


