import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ================= CONFIG =================

CSV_PATH = "/home/gondeleon/Full_v5 - GPS-AHRS-IMU/data_MOANA/gps.csv"        # <-- tu archivo
OUT_PATH = "/home/gondeleon/Full_v5 - GPS-AHRS-IMU/data_MOANA/gps_cut.csv"     # salida
OUT_HTML = "/home/gondeleon/Full_v5 - GPS-AHRS-IMU/data_MOANA/map.html"

# Rango temporal a exportar
START_TIME = "2024-03-11 06:57:52.432"
END_TIME   = "2024-03-11 07:00:00.000"

# --- Opcional: para que el mapa vaya fluido ---
DOWNSAMPLE_EVERY_N = 1   # 1 = no downsample. 5 = 1 de cada 5 puntos, etc.

# --- Importante: no unir con línea si hay saltos grandes ---
# Si el salto entre puntos es mayor a esto, se corta la línea (evita "spaghetti")
MAX_JUMP_METERS = 20.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia Haversine en metros (vectorizable)."""
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["timestamp", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["timestamp", "latitude", "longitude"]).copy()

    # Epoch -> datetime UTC
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Downsample para visualización
    if DOWNSAMPLE_EVERY_N and DOWNSAMPLE_EVERY_N > 1:
        df = df.iloc[::DOWNSAMPLE_EVERY_N].reset_index(drop=True)

    return df


def build_segmented_line(df: pd.DataFrame):
    """Devuelve lat/lon con NaNs para cortar la línea en saltos grandes."""
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    # Distancia entre puntos consecutivos
    d = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cut = d > MAX_JUMP_METERS

    lat_line = lat.astype(float).copy()
    lon_line = lon.astype(float).copy()

    # Insertar NaN "después" del punto donde hay salto
    # estrategia: crear arrays nuevos con NaNs insertados
    lat_out = [lat_line[0]]
    lon_out = [lon_line[0]]
    for i in range(1, len(lat_line)):
        if cut[i-1]:
            lat_out.append(np.nan)
            lon_out.append(np.nan)
        lat_out.append(lat_line[i])
        lon_out.append(lon_line[i])

    return np.array(lat_out), np.array(lon_out)


def make_map(df: pd.DataFrame):
    center = dict(
        lat=float(df["latitude"].median()),
        lon=float(df["longitude"].median())
    )

    hover = []
    hover.append("dt=%{customdata[0]}")
    hover.append("ts=%{customdata[1]:.3f}")
    if "yaw" in df.columns:   hover.append("yaw=%{customdata[2]}")
    if "roll" in df.columns:  hover.append("roll=%{customdata[3]}")
    if "pitch" in df.columns: hover.append("pitch=%{customdata[4]}")
    if "altitude" in df.columns: hover.append("alt=%{customdata[5]}")
    hovertemplate = "<br>".join(hover) + "<extra></extra>"

    # Customdata compacto (evita inflar el HTML)
    cols = ["datetime", "timestamp"]
    for c in ["yaw", "roll", "pitch", "altitude"]:
        if c in df.columns:
            cols.append(c)
    cd = df[cols].astype(str).to_numpy()

    lat_line, lon_line = build_segmented_line(df)

    fig = go.Figure()

    # Línea segmentada (con NaN para cortes)
    fig.add_trace(go.Scattermapbox(
        lat=lat_line,
        lon=lon_line,
        mode="lines",
        name="trajectory",
        hoverinfo="skip",
    ))

    # Puntos (hover con timestamp)
    fig.add_trace(go.Scattermapbox(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        name="points",
        marker=dict(size=6),
        customdata=cd,
        hovertemplate=hovertemplate,
    ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=center, zoom=14),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800,
        title="GNSS Trajectory (abre map.html en tu navegador)"
    )

    return fig


def export_subset(df: pd.DataFrame):
    if START_TIME is None or END_TIME is None:
        print("No export: START_TIME / END_TIME not set")
        return

    t0 = pd.to_datetime(START_TIME, utc=True)
    t1 = pd.to_datetime(END_TIME, utc=True)
    if t1 < t0:
        t0, t1 = t1, t0

    sub = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    if sub.empty:
        print("Subset vacío para ese rango.")
        return

    sub.drop(columns=["datetime"], inplace=True, errors="ignore")
    sub.to_csv(OUT_PATH, index=False)
    print(f"Exported subset: {OUT_PATH} | rows={len(sub)} | {t0} -> {t1}")


def main():
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print("from:", df["datetime"].iloc[0], "to:", df["datetime"].iloc[-1])

    fig = make_map(df)

    # En vez de fig.show() => HTML
    fig.write_html(OUT_HTML, auto_open=True)
    print(f"Wrote interactive map: {OUT_HTML}")

    export_subset(df)


if __name__ == "__main__":
    main()