import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# -----------------------------
# Cargar datos
# -----------------------------
df = pd.read_csv(input("Ruta de archivo.csv: "), parse_dates=["time"])
values = df["sst"].values.reshape(-1, 1)
print(f"Total de días: {len(values)}")

scaler = MinMaxScaler()
values = scaler.fit_transform(values)

# -----------------------------
# Configuración
# -----------------------------
dias_usados = input("Cantidad de días de entrada. Por defecto 30: ")
dias_to_predict = input("Días a predecir. Por defecto 7: ")
SEQ_LEN = int(dias_usados) if dias_usados != "" else 30        # días usados como entrada
PRED_LEN = int(dias_to_predict) if dias_to_predict != "" else 7           # días a predecir
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001


# -----------------------------
# Dataset temporal
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len - pred_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len:i+seq_len+pred_len])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TimeSeriesDataset(values, SEQ_LEN, PRED_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Modelo LSTM
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, PRED_LEN)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # último timestep
        return self.fc(out)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Entrenamiento
# -----------------------------
print("Entrenando modelo...")
for epoch in range(EPOCHS):
    loss_total = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss_total/len(loader):.6f}")

# -----------------------------
# Guardar modelo y scaler
# -----------------------------
joblib.dump(scaler, "scaler.save")
torch.save(model.state_dict(), "lstm_sst.pt")


print("✅ Modelo entrenado y guardado")
