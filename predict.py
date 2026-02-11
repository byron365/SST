import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error 

# -----------------------------
# Cargar datos
# -----------------------------
DATA_PATH = input("Ruta de archivo .csv: ")
df = pd.read_csv(DATA_PATH)
print("Procesando datos de entrenamiento...")

# 🔑 FORZAR conversión a datetime (evita el error)
df["time"] = pd.to_datetime(df["time"])

values = df["sst"].values.reshape(-1, 1)
print(f"Total de días: {len(values)}")
# -----------------------------
# Configuración
# -----------------------------
dias_usados = input("Cantidad de días de entrada. Por defecto 30: ")
dias_to_predict = input("Días a predecir. Por defecto 7: ")
SEQ_LEN = int(dias_usados) if dias_usados != "" else 30        # días usados como entrada
PRED_LEN = int(dias_to_predict) if dias_to_predict != "" else 7           # días a predecir
MODEL_PATH = "lstm_sst.pt"
SCALER_PATH = "scaler.save"




# -----------------------------
# Definición del modelo
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

# -----------------------------
# Cargar modelo
# -----------------------------
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -----------------------------
# Cargar scaler
# -----------------------------
scaler = joblib.load(SCALER_PATH)


# Normalizar con el mismo scaler del entrenamiento
values_scaled = scaler.transform(values)

# -----------------------------
# Datos reales recientes
# -----------------------------
real_recent = values[-SEQ_LEN:].flatten()
dates_real = df["time"].iloc[-SEQ_LEN:]

# -----------------------------
# Predicción
# -----------------------------
input_seq = torch.tensor(
    values_scaled[-SEQ_LEN:],
    dtype=torch.float32
).unsqueeze(0)  # (1, SEQ_LEN, 1)

with torch.no_grad():
    pred_scaled = model(input_seq).numpy()

# Volver a escala real
pred = scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).flatten()

# -----------------------------
# Fechas de predicción
# -----------------------------
last_date = dates_real.iloc[-1]

dates_pred = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=PRED_LEN
)

# -----------------------------
# Métricas
# -----------------------------
r_recent = df["sst"].values[-len(pred):]
mae = mean_absolute_error(r_recent, pred)
rmse = np.sqrt(mean_squared_error(r_recent, pred))

print("📊 Métricas del modelo:")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")

# -----------------------------
# Gráfica
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(dates_real, real_recent, label="Datos reales")
plt.plot(dates_pred, pred, label="Predicción", linestyle="--", marker="o")
plt.axvline(last_date, linestyle=":", label="Inicio predicción")

plt.xlabel("Fecha")
plt.ylabel("SST °C")
plt.title("SST real vs predicción")
plt.legend()
plt.grid(True)

# -------- FORMATO DE FECHAS --------
ax = plt.gca()

# Localizador automático (decide cuántas fechas mostrar)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

# Formato de fecha
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Rotar etiquetas
plt.xticks(rotation=45)

#MAE / RMSE
plt.text(
    0.02, 0.95,
    f"""
MAE: {mae:.2f}
RMSE: {rmse:.2f}
din: {SEQ_LEN}
dtp: {PRED_LEN}
    """,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.3)
)


plt.tight_layout()
plt.savefig("sst_real_vs_pred.png", dpi=300)
plt.close()

#Datos reales
plt.figure(figsize=(12, 6))

plt.plot(
    dates_real,
    real_recent,
    color="tab:blue",
    linewidth=2
)

plt.xlabel("Fecha")
plt.ylabel("SST (°C)")
plt.title("SST – Datos reales recientes")
plt.grid(alpha=0.3)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"sst_real_{SEQ_LEN}.png", dpi=300, bbox_inches="tight")
plt.close()

#Solo predicción
plt.figure(figsize=(12, 6))

plt.plot(
    dates_pred,
    pred,
    linestyle="--",
    marker="o",
    color="tab:orange",
    linewidth=2
)

plt.xlabel("Fecha")
plt.ylabel("SST (°C)")
plt.title(f"SST – Predicción a {PRED_LEN} días")
plt.grid(alpha=0.3)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"sst_pred_{PRED_LEN}.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Mostrar valores numéricos
# -----------------------------
print("📈 Predicción próximos días:")
for i, v in enumerate(pred, 1):
    print(f"Día +{i}: {v:.2f}")
