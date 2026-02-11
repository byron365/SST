import xarray as xr
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
DATA_DIR = Path(input("Coloca la ruta a los archivos .nc: "))
OUTPUT_CSV = "Data/sst_centroamerica.csv"

LAT_MIN, LAT_MAX = 5, 20
LON_MIN, LON_MAX = -100, -75

dfs = []

# ----------------------------
# PROCESAR ARCHIVOS
# ----------------------------
currentfile = 0
totalnc = len(sorted(DATA_DIR.glob("*.nc")))

print("Uniendo Archivos .nc a .csv")

for nc_file in sorted(DATA_DIR.glob("*.nc")):

    porcentaje = round((currentfile * 100)/totalnc,2)
    print(f"Procesando: {nc_file.name} - Porcentaje: {porcentaje}% - Restantes: {totalnc-currentfile}")
    currentfile += 1

    ds = xr.open_dataset(nc_file)

    # Variable SST 
    sst = ds["analysed_sst"]

    # Selección región
    sst_region = sst.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    # Promedio espacial
    sst_mean = sst_region.mean(dim=["lat", "lon"], skipna=True)

    # A DataFrame con nombre fijo
    df = sst_mean.to_dataframe(name="sst").reset_index()

    # Kelvin → Celsius
    df["sst"] = df["sst"] - 273.15

    dfs.append(df)

    ds.close()

# ----------------------------
# UNIR TODO
# ----------------------------
df_all = pd.concat(dfs, ignore_index=True)

# Quitar duplicados por tiempo
df_all = df_all.drop_duplicates(subset="time")

# Ordenar cronológicamente
df_all = df_all.sort_values("time")

# Guardar
df_all.to_csv(OUTPUT_CSV, index=False)

print(f"\nCSV generado en: {OUTPUT_CSV}, Primeras filas:")
print(df_all.head())

