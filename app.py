# app.py
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

import folium
import h3
import branca.colormap as cm
from streamlit_folium import st_folium

st.set_page_config(page_title="Hotspots delictivos (MVP)", layout="wide")

DATA_DIR = Path("outputs")
PRED_2023 = DATA_DIR / "predictions_2023.csv"
PRED_2024 = DATA_DIR / "predictions_2024.csv"
METRICS   = DATA_DIR / "metrics.json"
HEX_DIST  = Path("hex_pordistrito.csv")

CENTER_LIMA = [-12.05, -77.05]
INITIAL_ZOOM = 11


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # sanidad básica
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0.0)
    if "n_delitos" in df.columns:
        df["n_delitos"] = pd.to_numeric(df["n_delitos"], errors="coerce").fillna(0.0)
    if "hex_id" in df.columns:
        df["hex_id"] = df["hex_id"].astype(str)
    return df

@st.cache_data
def load_metrics(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_hex_distrito(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["hex_id","distrito_top","distritos_str"])
    hd = pd.read_csv(path)
    def main_dist(row):
        try:
            ds = ast.literal_eval(str(row.get("distritos", "[]")))
            ps = ast.literal_eval(str(row.get("pct_por_distrito", "[]")))
            if not ds or not ps:
                return np.nan
            i = int(np.argmax(ps))
            return ds[i]
        except Exception:
            return np.nan
    if "distrito_top" not in hd.columns:
        hd["distrito_top"] = hd.apply(main_dist, axis=1)
    if "distritos_str" not in hd.columns:
        def fmt(row):
            try:
                ds = ast.literal_eval(str(row.get("distritos", "[]")))
                ps = ast.literal_eval(str(row.get("pct_por_distrito", "[]")))
                return "; ".join([f"{d} ({p*100:.1f}%)" for d,p in zip(ds, ps)])
            except:
                return ""
        hd["distritos_str"] = hd.apply(fmt, axis=1)
    keep = ["hex_id","distrito_top","distritos_str"]
    hd["hex_id"] = hd["hex_id"].astype(str)
    return hd[keep].copy()

def clip_percentile(s: pd.Series, p: float):
    vmax = float(np.percentile(s, p)) if len(s) else 1.0
    return s.clip(lower=0, upper=vmax), vmax


def make_color_column(values: pd.Series, vmax: float) -> list[list[int]]:
    """Devuelve [[r,g,b,a], ...] como listas Python (no ndarrays)."""
    v = (values / max(vmax, 1e-9)).to_numpy().clip(0, 1)
    r = 255*(1-v) + 189*v
    g = 255*(1-v) +   0*v
    b = 178*(1-v) +  38*v
    a = np.full_like(r, 180.0)
    rgba = np.stack([r, g, b, a], axis=1)
    return np.rint(rgba).astype(int).tolist()

def clean_for_json(df: pd.DataFrame) -> list[dict]:
    d = df.copy()
    if "hex_id" in d.columns:
        d["hex_id"] = d["hex_id"].astype(str)
    d = d.where(pd.notnull(d), None)
    return d.to_dict(orient="records")

def make_deck(df: pd.DataFrame, value_col: str,
              tooltip_cols=("hex_id","distrito_top","distritos_str"),
              percentile=95, elevation_scale=50.0):
    view_state = pdk.ViewState(latitude=CENTER_LIMA[0], longitude=CENTER_LIMA[1],
                               zoom=INITIAL_ZOOM, pitch=40)
    if df.empty:
        return pdk.Deck(initial_view_state=view_state, map_style="light")

    vals = df[value_col].fillna(0.0)
    vals, vmax = clip_percentile(vals, percentile)
    colors = make_color_column(vals, vmax)

    data = df.copy()
    data["__color__"] = colors
    data["__elev__"]  = (vals * elevation_scale).astype(float)

    records = clean_for_json(data)

    layer = pdk.Layer(
        "H3HexagonLayer",
        data=records,
        get_hexagon="hex_id",
        get_fill_color="__color__",
        get_elevation="__elev__",
        elevation_scale=1,
        extruded=True,
        pickable=True,
        stroked=False,
        opacity=0.8,
    )

    # Evita formateadores tipo {:.1f} en tooltip (pydeck no los procesa)
    txt = "{hex_id}\n" + f"{value_col}: {{{value_col}}}"
    if tooltip_cols:
        for c in tooltip_cols:
            if c in df.columns:
                txt += f"\n{c}: {{{c}}}"

    return pdk.Deck(layers=[layer], initial_view_state=view_state,
                    tooltip={"text": txt}, map_style="light")


def _hex_boundary(hex_id: str):
    return h3.cell_to_boundary(hex_id)

def _add_layer(m, df, value_col, layer_name, vmax_percentile=95, tooltip_cols=None):
    """Agrega una capa choropleth a un mapa Folium."""
    if df.empty:
        return
    g = folium.FeatureGroup(name=layer_name, show=False)
    vals = df[value_col].clip(lower=0)
    vmax = float(np.percentile(vals, vmax_percentile)) if len(vals) else 1.0
    vmin = 0.0
    cmap = cm.LinearColormap(
        ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=vmin, vmax=vmax
    )
    cmap.caption = f"{layer_name} ({value_col})"

    for _, row in df.iterrows():
        hex_id = str(row["hex_id"])
        v = float(max(0, row[value_col]))
        color = cmap(min(v, vmax))
        boundary = _hex_boundary(hex_id)
        tip_parts = [f"{value_col}: {v:.1f}", f"hex: {hex_id}"]
        if tooltip_cols:
            for c in tooltip_cols:
                if c in row and pd.notna(row[c]):
                    tip_parts.append(f"{c}: {row[c]}")
        tooltip_html = "<br>".join(tip_parts)
        folium.Polygon(
            locations=boundary,
            weight=0.6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=tooltip_html,
        ).add_to(g)
    g.add_to(m)
    cmap.add_to(m)

def build_folium_map(df23_raw: pd.DataFrame, df24_raw: pd.DataFrame,
                     top_n=None, vmax_percentile=95, tiles="CartoDB positron"):
    # Filtra años
    real23 = df23_raw[["hex_id","anio","n_delitos"]].copy() if "n_delitos" in df23_raw.columns else pd.DataFrame(columns=["hex_id","anio","n_delitos"])
    pred23 = df23_raw[["hex_id","anio","y_pred"]].copy() if "y_pred" in df23_raw.columns else pd.DataFrame(columns=["hex_id","anio","y_pred"])
    pred24 = df24_raw[["hex_id","anio","y_pred"]].copy() if "y_pred" in df24_raw.columns else pd.DataFrame(columns=["hex_id","anio","y_pred"])

    real23 = real23[real23["anio"]==2023].copy()
    pred23 = pred23[pred23["anio"]==2023].copy()
    pred24 = pred24[pred24["anio"]==2024].copy()

    def _keep_top(d, col):
        if d.empty:
            return d
        if top_n is None:
            return d
        return d.sort_values(col, ascending=False).head(top_n).copy()

    real23 = _keep_top(real23, "n_delitos")
    pred23 = _keep_top(pred23, "y_pred")
    pred24 = _keep_top(pred24, "y_pred")

    m = folium.Map(location=CENTER_LIMA, zoom_start=INITIAL_ZOOM, tiles=tiles)
    _add_layer(m, real23, "n_delitos", "2023 REAL (n_delitos)", vmax_percentile=vmax_percentile)
    _add_layer(m, pred23, "y_pred", "2023 PRED (y_pred)", vmax_percentile=vmax_percentile)
    _add_layer(m, pred24, "y_pred", "2024 PRED (y_pred)", vmax_percentile=vmax_percentile)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


pred23 = load_csv(PRED_2023)
pred24 = load_csv(PRED_2024)
metrics = load_metrics(METRICS)
hexdist = load_hex_distrito(HEX_DIST)

if not pred23.empty:
    pred23 = pred23.merge(hexdist, on="hex_id", how="left")
if not pred24.empty:
    pred24 = pred24.merge(hexdist, on="hex_id", how="left")


st.title("Predicción de criminalidad – MVP (Lima)")

# Métricas de modelo
colA, colB, colC, colD = st.columns(4)
if metrics:
    m_te = metrics.get("metrics", {}).get("test", {})
    colA.metric("MAE (test)", f"{m_te.get('MAE', np.nan):.1f}")
    colB.metric("RMSE (test)", f"{m_te.get('RMSE', np.nan):.1f}")
    colC.metric("R² (test)", f"{m_te.get('R2', np.nan):.3f}")
    colD.metric("MAPE% (test)", f"{m_te.get('MAPE%', np.nan):.1f}")
else:
    colA.info("Sin metrics.json")

st.markdown("---")

# Sidebar (controles globales)
st.sidebar.header("Controles")
engine = st.sidebar.radio("Motor de mapa", options=["Pydeck (rápido)","Folium (capas)"], index=0)
year_opt = st.sidebar.radio("Año", options=["2023 (real/pred)","2024 (pred)"], index=0)
metric_opt = st.sidebar.selectbox("Métrica a visualizar",
                                  options=["y_pred","n_delitos"],
                                  index=0 if year_opt!="2023 (real/pred)" else 0)
top_n = st.sidebar.slider("Top-N hex a mostrar", min_value=100, max_value=5000, value=1200, step=100)
pct = st.sidebar.slider("Recorte de color (percentil)", min_value=80, max_value=100, value=95, step=1)
elev = st.sidebar.slider("Escala de elevación (3D)", min_value=0.0, max_value=200.0, value=60.0, step=5.0)
# Filtro por distrito principal
all_dists = sorted(pd.Series(pd.concat([
    pred23.get("distrito_top", pd.Series(dtype=str)),
    pred24.get("distrito_top", pd.Series(dtype=str))
], ignore_index=True).dropna().unique()).tolist())
dist_sel = st.sidebar.selectbox("Filtrar distrito (opcional)", options=["(todos)"] + all_dists, index=0)

# =========================
# Selección por año
# =========================
if year_opt == "2023 (real/pred)":
    df_map = pred23[pred23["anio"] == 2023].copy()
    if metric_opt not in df_map.columns:
        metric_opt = "y_pred"
        st.sidebar.warning("No hay 'n_delitos' en 2023; usando 'y_pred'.")
else:
    df_map = pred24[pred24["anio"] == 2024].copy()
    metric_opt = "y_pred"

if dist_sel != "(todos)" and "distrito_top" in df_map.columns:
    df_map = df_map[df_map["distrito_top"] == dist_sel]

if not df_map.empty:
    df_map = df_map.sort_values(metric_opt, ascending=False).head(top_n).copy()


if engine == "Pydeck (rápido)":
    deck = make_deck(df_map, value_col=metric_opt, percentile=pct, elevation_scale=elev)
    st.pydeck_chart(deck, use_container_width=True)
else:
    folium_map = build_folium_map(pred23, pred24, top_n=top_n, vmax_percentile=pct, tiles="CartoDB positron")
    st_folium(folium_map, width=None, height=720)

st.markdown("### Rankings")
t1, t2 = st.columns(2)
with t1:
    st.caption(f"Top {min(top_n, 25)} celdas H3 por **{metric_opt}**")
    if not df_map.empty:
        cols = ["hex_id", metric_opt]
        for c in ["distrito_top","distritos_str","anio","year_total","n_delitos","y_pred"]:
            if c in df_map.columns and c not in cols:
                cols.append(c)
        st.dataframe(df_map.sort_values(metric_opt, ascending=False).head(min(25, len(df_map)))[cols])
    else:
        st.info("Sin datos para los filtros seleccionados.")
with t2:
    st.caption(f"Top 15 distritos por **{metric_opt}** (suma de hex)")
    if not df_map.empty and "distrito_top" in df_map.columns:
        top_dist = (df_map.groupby("distrito_top")[metric_opt]
                    .sum().reset_index().sort_values(metric_opt, ascending=False).head(15))
        top_dist = top_dist.rename(columns={"distrito_top":"distrito"})
        st.dataframe(top_dist)
    else:
        st.info("No hay distritos disponibles para esta vista.")

st.markdown("### Descargas")
c1, c2, c3 = st.columns(3)
with c1:
    if PRED_2023.exists():
        st.download_button("Descargar CSV 2023", data=PRED_2023.read_bytes(), file_name="predictions_2023.csv")
    else:
        st.button("Falta predictions_2023.csv", disabled=True)
with c2:
    if PRED_2024.exists():
        st.download_button("Descargar CSV 2024", data=PRED_2024.read_bytes(), file_name="predictions_2024.csv")
    else:
        st.button("Falta predictions_2024.csv", disabled=True)
with c3:
    if METRICS.exists():
        st.download_button("Descargar metrics.json", data=METRICS.read_bytes(), file_name="metrics.json")
    else:
        st.button("Sin metrics.json", disabled=True)

st.markdown("---")
st.caption("Nota: H3HexagonLayer (Deck.gl) para rendimiento; Folium para capas conmutables (real/pred). "
           "Los valores se recortan por percentil para mejorar el contraste visual.")
