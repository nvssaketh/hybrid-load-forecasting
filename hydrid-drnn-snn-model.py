# STEP 1: Install and import necessary libraries
!pip install snntorch -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully!")

# STEP 2: Load and preprocess the data
def load_and_preprocess_data():
    df_raw = pd.read_csv('hourly_load_2016.csv', header=None, names=['datetime', 'load'])
    # Remove rows with missing or invalid load values
    df_raw = df_raw.dropna()
    df_raw = df_raw[df_raw['load'] > 0]

    loads = df_raw['load'].values.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_loads = scaler.fit_transform(loads).flatten()

    seq_len = 24  # Use past 24 hours to predict next hour

    X, y = [], []
    for i in range(len(scaled_loads) - seq_len):
        X.append(scaled_loads[i:i+seq_len])
        y.append(scaled_loads[i+seq_len])

    X, y = np.array(X), np.array(y)

    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Data loaded: {len(X)} samples, train: {len(X_train)}, test: {len(X_test)}")
    return (X_train, y_train), (X_test, y_test), scaler

# STEP 3: Define Dilated LSTM cell for DRNN
class DilatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.dilation = dilation
        self.hidden_size = hidden_size

    def forward(self, x_seq):
        # x_seq shape: (seq_len, batch, input_size)
        seq_len, batch_size, _ = x_seq.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        outputs = []

        for t in range(seq_len):
            if t % self.dilation == 0:
                h_t, c_t = self.lstm(x_seq[t], (h_t, c_t))
            outputs.append(h_t.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs[-1]  # Return last output

# STEP 4: Build DRNN model using dilated LSTM layers
class DRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, dilation_rates=[1,2,4]):
        super().__init__()
        self.cells = nn.ModuleList([
            DilatedLSTMCell(input_size if i==0 else hidden_size, hidden_size, dilation)
            for i, dilation in enumerate(dilation_rates)
        ])
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(1,0,2)  # (seq_len, batch, input_size)
        for cell in self.cells:
            x_out = cell(x)
            x = x_out.unsqueeze(0).repeat(x.size(0),1,1)  # Broadcast for next layer
        out = self.fc(x_out)
        return out.squeeze()

# STEP 5: Define optimized SNN model (similar to provided example)
class OptimizedSNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        outputs = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            out = self.fc2(spk1)
            outputs.append(out)
        return torch.stack(outputs).mean(0).squeeze()

# STEP 6: Train DRNN model
def train_drnn_model(model, X_train, y_train, epochs=50, batch_size=64):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
                                             torch.tensor(y_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(loader.dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"DRNN Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    print("DRNN training complete.")
    return model

# STEP 7: Train SNN model
def train_snn_model(X_train, y_train, epochs=50, batch_size=128):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = OptimizedSNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out.unsqueeze(1), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(loader.dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"SNN Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    print("SNN training complete.")
    return model

# STEP 8: Make predictions and create hybrid output
def hybrid_predict(drnn_model, snn_model, X_test, alpha=0.6):
    drnn_model.eval()
    snn_model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        drnn_preds = drnn_model(X_test_tensor.unsqueeze(-1)).cpu().numpy()
        snn_preds = snn_model(X_test_tensor).cpu().numpy()

    min_len = min(len(drnn_preds), len(snn_preds))
    hybrid_preds = alpha * drnn_preds[:min_len] + (1 - alpha) * snn_preds[:min_len]
    return drnn_preds[:min_len], snn_preds[:min_len], hybrid_preds

# STEP 9: Evaluate model performance and plot results
def evaluate_performance(y_true, preds, model_name):
    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return mse, mae, mape

def plot_results(y_true_orig, drnn_pred_orig, snn_pred_orig, hybrid_pred_orig):
    plt.figure(figsize=(15,7))
    n = 400
    plt.plot(y_true_orig[:n], label='Actual', linewidth=2)
    plt.plot(drnn_pred_orig[:n], label='DRNN Prediction', linestyle='--')
    plt.plot(snn_pred_orig[:n], label='SNN Prediction', linestyle=':')
    plt.plot(hybrid_pred_orig[:n], label='Hybrid Prediction', linewidth=2)
    plt.title("DRNN-SNN Hybrid Electricity Load Forecasting")
    plt.xlabel("Time (hours)")
    plt.ylabel("Load")
    plt.legend()
    plt.grid(True)
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), scaler = load_and_preprocess_data()

    # Train DRNN
    drnn_model = DRNNModel()
    drnn_model = train_drnn_model(drnn_model, X_train, y_train)

    # Train SNN
    snn_model = train_snn_model(X_train, y_train)

    # Predict
    drnn_preds, snn_preds, hybrid_preds = hybrid_predict(drnn_model, snn_model, X_test)

    # Inverse scale predictions and true values
    y_test_orig = scaler.inverse_transform(y_test[:len(hybrid_preds)].reshape(-1,1)).flatten()
    drnn_pred_orig = scaler.inverse_transform(drnn_preds.reshape(-1,1)).flatten()
    snn_pred_orig = scaler.inverse_transform(snn_preds.reshape(-1,1)).flatten()
    hybrid_pred_orig = scaler.inverse_transform(hybrid_preds.reshape(-1,1)).flatten()

    # Evaluate
    evaluate_performance(y_test_orig, drnn_pred_orig, "DRNN")
    evaluate_performance(y_test_orig, snn_pred_orig, "SNN")
    evaluate_performance(y_test_orig, hybrid_pred_orig, "Hybrid DRNN-SNN")

    # Plot
    plot_results(y_test_orig, drnn_pred_orig, snn_pred_orig, hybrid_pred_orig)
