import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pydeck as pdk
import altair as alt
import glob
import math
import os


class ResBlockMLP(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, hidden) if in_ch != hidden else None
    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        res = self.res_proj(x) if self.res_proj is not None else x
        return self.act(h + res)

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, A_norm):
        super().__init__()
        self.A = A_norm
        self.lin = nn.Linear(in_feat, out_feat)
    def forward(self, x):
        B, S, N, F = x.shape
        x_in = x.reshape(B * S, N, F)
        x_prop = torch.matmul(self.A, x_in)
        out = self.lin(x_prop).reshape(B, S, N, -1)
        return torch.relu(out)

class TemporalAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
    def forward(self, x):
        B, S, N, F = x.shape
        q = self.query(x).permute(0, 2, 1, 3).reshape(B*N, S, F)
        k = self.key(x).permute(0, 2, 1, 3).reshape(B*N, S, F)
        v = self.value(x).permute(0, 2, 1, 3).reshape(B*N, S, F)
        att = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(F), dim=-1)
        return torch.bmm(att, v).sum(dim=1).reshape(B, N, F)

class IntegratedModel(nn.Module):
    def __init__(self, n_nodes, seq_len, in_ch, hidden, A_norm):
        super().__init__()
        self.resblock = ResBlockMLP(in_ch, hidden)
        self.gcn = SimpleGCNLayer(hidden, hidden, A_norm)
        self.lstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.att = TemporalAttention(hidden)
        self.out_proj = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

    def forward(self, seq):
        x = self.resblock(seq.unsqueeze(-1))
        x = self.gcn(x)
        B, S, N, F = x.shape
        x_perm = x.permute(0, 2, 1, 3)
        x_resh = x_perm.reshape(B*N, S, F)
        lstm_out, _ = self.lstm(x_resh)
        lstm_out = lstm_out.reshape(B, N, S, F).permute(0, 2, 1, 3)
        y = self.out_proj(self.att(lstm_out)).squeeze(-1)
        return y

# ==============================================================================
# 2. CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="S.P.C. - Sistema Policial", layout="wide", page_icon="🚔")

st.markdown("""
<style>
    .metric-container {background-color: #f0f2f6; border-radius: 10px; padding: 10px; border-left: 5px solid #ff4b4b;}
    h1 {color: #0e1117;}
    .legend-box {
        background-color: rgba(20, 20, 20, 0.8); 
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-size: 0.9em;
        color: white;
    }
    .legend-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "ig_outputs/integrated_model.pt"
DATA_GLOB = "./data/final*.csv"
DEFAULT_LAT = -12.046374
DEFAULT_LON = -77.042793

@st.cache_resource
def load_system():
    if not os.path.exists(MODEL_PATH): return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint['cfg']
    dist_names = checkpoint['district_names']
    A_norm = torch.tensor(checkpoint['A_norm'], dtype=torch.float32)
    model = IntegratedModel(len(dist_names), cfg['seq_len'], 1, cfg['hidden_dim'], A_norm)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return {
        "model": model, "districts": dist_names,
        "scaler_mean": checkpoint['scaler_mean'], "scaler_scale": checkpoint['scaler_scale'],
        "seq_len": cfg['seq_len']
    }

@st.cache_data
def load_historical_data(districts_order):
    files = sorted(glob.glob(DATA_GLOB))
    if not files: return None, None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = ["anio", "X", "Y", "distrito"]
    df = df.dropna(subset=cols)
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype(int)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["distrito"] = df["distrito"].astype(str).str.strip().str.upper()
    
    centroids = df.groupby("distrito")[["X", "Y"]].mean().reset_index()
    centroids = centroids.set_index("distrito").reindex(districts_order).reset_index()
    centroids = centroids.fillna({"X": DEFAULT_LON, "Y": DEFAULT_LAT})
    
    pivot = df.groupby(["anio", "distrito"]).size().reset_index(name="n_delitos")
    pivot = pivot.pivot(index="anio", columns="distrito", values="n_delitos").fillna(0)
    pivot = pivot.reindex(columns=districts_order, fill_value=0)
    return pivot, centroids

# ==============================================================================
# 3. INFERENCIA Y LÓGICA
# ==============================================================================
system = load_system()
if not system:
    st.error(f"❌ Error: No se encontró el modelo en '{MODEL_PATH}'.")
    st.stop()
history_df, centroids_df = load_historical_data(system["districts"])

# HEADER
col_logo, col_title = st.columns([1, 6])
with col_logo: st.markdown("# 🚔") 
with col_title:
    st.title("Sistema de Predicción de Criminalidad (S.P.C.)")
    st.markdown("**Unidad de Análisis Táctico - Modelo GCN Espacio-Temporal**")

# SIDEBAR
st.sidebar.header("🎛️ Panel de Control")

with st.sidebar.expander("🛠️ Simulador de Escenarios", expanded=False):
    st.info("Modifica la historia reciente para ver impacto futuro.")
    use_simulation = st.toggle("Activar Simulación")
    last_year = int(history_df.index.max())
    future_year = last_year + 1
    
    if use_simulation:
        dist_sim = st.selectbox("Distrito a intervenir", system["districts"])
        val_real = int(history_df.iloc[-1][dist_sim])
        val_sim = st.number_input(f"Incidentes en {dist_sim}", value=val_real)
        working_df = history_df.copy()
        working_df.at[last_year, dist_sim] = val_sim
    else:
        working_df = history_df

st.sidebar.subheader("Filtros de Análisis")
crime_range = st.sidebar.slider(
    "Rango de Incidentes Permitido",
    min_value=0, max_value=5000, value=(100, 5000),
    help="Muestra solo distritos cuya proyección caiga dentro de este rango."
)
selected_districts = st.sidebar.multiselect(
    "Filtrar Distritos Específicos",
    options=system["districts"],
    default=[]
)

map_pitch = st.sidebar.slider("Inclinación Mapa 3D", 0, 60, 45)

# Valores fijos visuales (Sin sliders en el UI)
elevation_scale = 5
bar_radius = 200

# CALCULO PREDICCION
seq_len = system["seq_len"]
input_window = working_df.iloc[-seq_len:].values
input_scaled = (input_window - system["scaler_mean"]) / system["scaler_scale"]
tensor_in = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    pred_scaled = system["model"](tensor_in).numpy()[0]

pred_raw = np.maximum((pred_scaled * system["scaler_scale"]) + system["scaler_mean"], 0)

results = pd.DataFrame({
    "Distrito": system["districts"],
    "Proyeccion": pred_raw,
    "Real_Anterior": working_df.iloc[-1].values,
    "lat": centroids_df["Y"].values,
    "lon": centroids_df["X"].values
})

# CORRECCIÓN AQUÍ: Unificamos a "Diferencia"
results["Diferencia"] = results["Proyeccion"] - results["Real_Anterior"]
results["Tendencia"] = np.where(results["Diferencia"] > 0, "⬆️ Sube", "⬇️ Baja")

def clasificar_riesgo(val):
    if val < 500: return "Bajo 🟢"
    elif val < 1500: return "Medio 🟡"
    else: return "Alto 🔴"

results["Nivel_Riesgo"] = results["Proyeccion"].apply(clasificar_riesgo)

# KPI
k1, k2, k3, k4 = st.columns(4)
total_proj = results["Proyeccion"].sum()
delta_total = ((total_proj - results["Real_Anterior"].sum()) / results["Real_Anterior"].sum()) * 100
top_risk = results.loc[results["Proyeccion"].idxmax()]

k1.metric("Proyección Total", f"{total_proj:,.0f}", f"{delta_total:.1f}%")
k2.metric("Distrito Más Crítico", top_risk["Distrito"])
k3.metric("Riesgo Máximo", f"{top_risk['Proyeccion']:.0f}")
k4.metric("Año Objetivo", f"{future_year}")

# TABS
tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Táctico", "📊 Comparativa", "📥 Datos"])

# --- TAB 1: MAPA TÁCTICO ---
with tab1:
    # Aplicar Filtros
    map_data = results[
        (results["Proyeccion"] >= crime_range[0]) & 
        (results["Proyeccion"] <= crime_range[1])
    ].copy()
    
    if selected_districts:
        map_data = map_data[map_data["Distrito"].isin(selected_districts)]

    st.markdown(f"**Visualizando {len(map_data)} distritos**")
    
    st.markdown("""
    <div class="legend-box">
        <span class="legend-dot" style="background-color:rgb(50, 50, 50)"></span> Bajo
        <span class="legend-dot" style="background-color:rgb(120, 100, 50)"></span> Medio
        <span class="legend-dot" style="background-color:rgb(255, 0, 50)"></span> Crítico (Alto Riesgo)
    </div>
    """, unsafe_allow_html=True)
    
    if map_data.empty:
        st.warning("⚠️ No hay datos con los filtros actuales.")
    else:
        # Pre-formatear strings para evitar el bug del Tooltip
        map_data["txt_proy"] = map_data["Proyeccion"].apply(lambda x: f"{x:,.0f}")
        map_data["txt_ant"] = map_data["Real_Anterior"].apply(lambda x: f"{x:,.0f}")
        
        # Color Dinámico
        max_val = map_data["Proyeccion"].max()
        def get_color(val):
            ratio = val / (max_val + 1e-5)
            return [int(255 * ratio), int(255 * (1 - ratio)), 50, 200]
        map_data["color"] = map_data["Proyeccion"].apply(get_color)
        
        # CAPA 3D
        layer = pdk.Layer(
            "ColumnLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_elevation="Proyeccion",
            elevation_scale=elevation_scale,
            radius=bar_radius,
            get_fill_color="color",
            pickable=True,
            extruded=True,
            auto_highlight=True,
        )
        
        view_state = pdk.ViewState(
            latitude=map_data["lat"].mean(), 
            longitude=map_data["lon"].mean(), 
            zoom=10, 
            pitch=map_pitch
        )
        
        tooltip_html = {
            "html": "<b>{Distrito}</b><br/>"
                    "🚨 Proyección: <b>{txt_proy}</b><br/>"
                    "⚠️ Riesgo: {Nivel_Riesgo}<br/>"
                    "🔙 Anterior: {txt_ant}<br/>"
                    "📈 {Tendencia}",
            "style": {"backgroundColor": "#1f2937", "color": "white", "zIndex": "1000"}
        }
        
        st.pydeck_chart(pdk.Deck(
            map_style="dark", 
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip_html
        ), use_container_width=True)

# --- TAB 2: COMPARATIVA ---
with tab2:
    st.subheader("Realidad vs Proyección")
    
    chart_data = results.copy()
    if selected_districts:
        chart_data = chart_data[chart_data["Distrito"].isin(selected_districts)]
    
    # Mostrar Top 20 si hay muchos
    chart_data = chart_data.sort_values("Proyeccion", ascending=False).head(20)
    
    df_long = pd.melt(chart_data, id_vars=["Distrito"], value_vars=["Real_Anterior", "Proyeccion"],
                      var_name="Escenario", value_name="Incidentes")
    
    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('Distrito:N', sort='-y', title=None),
        y=alt.Y('Incidentes:Q', title='N° Incidentes'),
        color=alt.Color('Escenario:N', scale=alt.Scale(range=['#9ca3af', '#ef4444']), legend=alt.Legend(title="Tipo")),
        tooltip=['Distrito', 'Escenario', alt.Tooltip('Incidentes', format=',.0f')],
        xOffset='Escenario:N'
    ).properties(height=450)
    
    st.altair_chart(chart, use_container_width=True)
    st.info("ℹ️ Haz clic en los tres puntos (...) arriba del gráfico para guardar como imagen.")

# --- TAB 3: DATOS ---
with tab3:
    st.subheader("Datos Operativos")
    
    # CORRECCIÓN: Usamos 'Diferencia' que sí existe
    st.dataframe(
        results[["Distrito", "Nivel_Riesgo", "Real_Anterior", "Proyeccion", "Diferencia", "Tendencia"]]
        .sort_values("Proyeccion", ascending=False)
        .style.background_gradient(cmap="Reds", subset=["Proyeccion"])
        .format({"Proyeccion": "{:.0f}", "Real_Anterior": "{:.0f}", "Diferencia": "{:+.0f}"}),
        use_container_width=True
    )
    
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.download_button(
            "📥 Descargar CSV Táctico", 
            data=results.to_csv(index=False).encode('utf-8'), 
            file_name=f"prediccion_{future_year}.csv", 
            mime='text/csv'
        )
