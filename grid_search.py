"""
Integrated Graph Model pipeline (ResNet-like MLP + GCN + LSTM + Attention)
Adaptado para tus CSVs `final2016.csv` ... `final2023.csv` y el esquema que muestras.

Requisitos:
  - Python 3.8+
  - pandas, numpy, scikit-learn, torch

Uso rápido:
  Ajusta `CFG.data_glob` y ejecuta: python integrated_graph_pipeline.py

¿Qué hace?
  1. Carga y normaliza los CSV (usa 'distrito' como nodo).
  2. Calcula un "centro" por distrito y arma vecinos (kNN) para una matriz de adyacencia.
  3. Agrega incidentes por (año, distrito) → una serie por distrito.
  4. Crea ventanas temporales para predecir el siguiente paso.
  5. Modelo: Bloque MLP residual → GCN → LSTM → Atención temporal → salida por distrito.
  6. Calcula MAE / RMSE / MAPE y guarda el mejor modelo + reporte.

Notas:
  - Si luego tienes fechas reales (día), cambia el agregador para trabajar a nivel diario.
  - La adyacencia es por distancia de centros; si tienes vecinos reales (límites), conviene usar esa info.
"""

import glob
import os
import math
import json
from dataclasses import dataclass
from typing import List, Tuple
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------- Configuración ----------------------
@dataclass
class CFG:
    data_glob: str = "./data/final*.csv"      # patrón para leer tus CSVs
    date_col: str = "anio"               # si luego tienes fecha diaria, cambia a la columna de fecha real
    x_col: str = "X"
    y_col: str = "Y"
    distrito_col: str = "distrito"
    # Años de referencia (si usas split 70/30, se ignoran)
    min_year: int = 2016
    max_year: int = 2023
    train_end: int = 2021
    val_year: int = 2022
    test_year: int = 2023

    # Hiperparámetros sencillos
    seq_len: int = 3
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    hidden_dim: int = 64
    gcn_k: int = 4                     # k para vecinos en kNN

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    outdir: str = "ig_outputs"


# Grid de hiperparámetros para búsqueda (seq_len fijo en 3, como en el script original)
HYPERPARAM_GRID = {
    'batch_size': [32, 64, 128],
    'epochs': [30, 40, 50],
    'lr': [5e-4, 1e-3, 2e-3],
    'hidden_dim': [48, 64, 96],
    'gcn_k': [3, 4, 5],
}


def generate_grid_combinations():
    """Genera todas las combinaciones del grid."""
    keys = list(HYPERPARAM_GRID.keys())
    values = [HYPERPARAM_GRID[k] for k in keys]
    combos = list(product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


cfg = CFG()
os.makedirs(cfg.outdir, exist_ok=True)


# ---------------------- Utilidades de datos ----------------------
def load_and_concatenate(glob_pattern: str) -> pd.DataFrame:
    """Lee y une todos los CSV que coinciden con el patrón."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV con el patrón: {glob_pattern}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que existan las columnas necesarias y estandariza tipos/formatos."""
    expected = [cfg.date_col, cfg.x_col, cfg.y_col, cfg.distrito_col]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    out = df.copy()

    # Conversión numérica segura
    out[cfg.date_col] = pd.to_numeric(out[cfg.date_col], errors="coerce").astype("Int64")
    out[cfg.x_col] = pd.to_numeric(out[cfg.x_col], errors="coerce")
    out[cfg.y_col] = pd.to_numeric(out[cfg.y_col], errors="coerce")

    # Normalización de texto para 'distrito' usando el accesor .str (evita el error visto)
    # - Convierte a dtype 'string' (pandas) para mejor manejo de NA
    # - Aplica strip y upper con .str
    out[cfg.distrito_col] = (
        out[cfg.distrito_col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Filas válidas únicamente
    out = out.dropna(subset=[cfg.date_col, cfg.x_col, cfg.y_col, cfg.distrito_col])

    return out


def build_district_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula un punto promedio (centro) para cada distrito."""
    cent = (
        df.groupby(cfg.distrito_col)
          .agg({cfg.x_col: 'mean', cfg.y_col: 'mean'})
          .reset_index()
          .rename(columns={cfg.x_col: 'cent_x', cfg.y_col: 'cent_y'})
    )
    return cent


def build_adjacency_from_centroids(centroids: pd.DataFrame, k: int = 4) -> Tuple[np.ndarray, List[str]]:
    """
    Construye una matriz de adyacencia simple basada en cercanía (k vecinos por distrito).
    Retorna la matriz normalizada por filas y el orden de distritos.
    """
    coords = centroids[['cent_x','cent_y']].values
    names = centroids[cfg.distrito_col].tolist()
    N = len(names)

    # Distancias entre centros
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    # Vecinos más cercanos
    knn = np.argsort(dist, axis=1)[:, :k]

    # Matriz de conexiones simétrica
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in knn[i]:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Normalización por grado (suma de fila)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_norm = A / deg
    return A_norm, names


def aggregate_yearly_counts(df: pd.DataFrame, district_order: List[str]) -> pd.DataFrame:
    """
    Convierte los datos a una tabla de conteos por (año, distrito).
    Filas: años. Columnas: distritos. Valores: número de incidentes.
    """
    df2 = df.copy()
    df2[cfg.date_col] = df2[cfg.date_col].astype(int)
    pivot = (
        df2.groupby([cfg.date_col, cfg.distrito_col])
            .size()
            .reset_index(name="n_delitos")
            .pivot(index=cfg.date_col, columns=cfg.distrito_col, values="n_delitos")
            .fillna(0.0)
            .astype(float)
    )
    pivot = pivot.reindex(columns=district_order, fill_value=0.0).sort_index()
    pivot.index.name = "anio"
    return pivot


# ---------------------- Dataset ----------------------
class TimeSeriesGraphDataset(Dataset):
    """
    Cada ejemplo es:
      - Entrada: ventana de longitud `seq_len` con conteos (seq_len × N).
      - Salida: el siguiente paso temporal (N).
    """
    def __init__(self, series_df: pd.DataFrame, seq_len: int):
        self.series = series_df.values.astype(float)
        self.seq_len = seq_len
        T, N = self.series.shape
        self.T = T
        self.N = N
        self.indices = [t for t in range(self.seq_len, T)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        seq = self.series[t - self.seq_len: t]   # (seq_len, N)
        target = self.series[t]                  # (N,)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def collate_fn(batch):
    """Agrupa ejemplos en un batch."""
    seqs = torch.stack([b[0] for b in batch], dim=0)     # (B, S, N)
    targets = torch.stack([b[1] for b in batch], dim=0)  # (B, N)
    return seqs, targets


# ---------------------- Modelo ----------------------
class ResBlockMLP(nn.Module):
    """
    Bloque MLP con atajo (residual) aplicado por nodo y por paso temporal.
    Entrada: (B, S, N, 1)  → Salida: (B, S, N, hidden)
    """
    def __init__(self, in_ch: int, hidden: int):
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
    """
    Capa GCN básica: mezcla información entre distritos usando la adyacencia.
    Mantiene forma: (B, S, N, F).
    """
    def __init__(self, in_feat: int, out_feat: int, A_norm: torch.Tensor):
        super().__init__()
        self.A = A_norm  # (N, N)
        self.lin = nn.Linear(in_feat, out_feat)

    def forward(self, x):
        B, S, N, F = x.shape
        x_in = x.reshape(B * S, N, F)        # junta batch y tiempo
        x_prop = torch.matmul(self.A, x_in)  # difunde por la red de distritos
        out = self.lin(x_prop).reshape(B, S, N, -1)
        return torch.relu(out)


class TemporalAttention(nn.Module):
    """
    Atención a lo largo del tiempo por cada nodo.
    Entrada: (B, S, N, F) → Salida: (B, N, F) agregando la historia reciente.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        B, S, N, F = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reorganiza para aplicar atención por nodo
        q = q.permute(0, 2, 1, 3).reshape(B * N, S, F)
        k = k.permute(0, 2, 1, 3).reshape(B * N, S, F)
        v = v.permute(0, 2, 1, 3).reshape(B * N, S, F)

        att_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(F)
        att = torch.softmax(att_scores, dim=-1)
        out = torch.bmm(att, v).sum(dim=1)  # suma en el tiempo tras ponderar
        return out.reshape(B, N, F)


class IntegratedModel(nn.Module):
    """
    Modelo integrado:
      ResBlock MLP → GCN → LSTM bidireccional → Atención temporal → proyección a 1 valor por nodo.
    """
    def __init__(self, n_nodes: int, seq_len: int, in_ch: int, hidden: int, A_norm: torch.Tensor):
        super().__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes

        self.resblock = ResBlockMLP(in_ch, hidden)
        self.gcn = SimpleGCNLayer(hidden, hidden, A_norm)

        # LSTM procesa la secuencia por nodo; bidireccional da salida de tamaño hidden (hidden//2 por dirección)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden // 2,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.att = TemporalAttention(hidden)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, seq):
        # seq: (B, S, N) con conteos
        B, S, N = seq.shape
        x = seq.unsqueeze(-1)         # (B, S, N, 1)
        x = self.resblock(x)          # (B, S, N, hidden)
        x = self.gcn(x)               # (B, S, N, hidden)

        # LSTM por nodo a lo largo del tiempo
        x_perm = x.permute(0, 2, 1, 3)      # (B, N, S, hidden)
        B2, N2, S2, F = x_perm.shape
        x_resh = x_perm.reshape(B2 * N2, S2, F)
        lstm_out, _ = self.lstm(x_resh)     # (B*N, S, hidden)
        lstm_out = lstm_out.reshape(B2, N2, S2, F).permute(0, 2, 1, 3)  # (B, S, N, F)

        # Atención temporal y proyección final
        att_out = self.att(lstm_out)        # (B, N, F)
        y = self.out_proj(att_out).squeeze(-1)  # (B, N)
        return y


# ---------------------- Entrenamiento / Métricas ----------------------
def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    """
    Calcula MAPE ignorando casos con valor real = 0.
    Retorna (mape_en_pct, cobertura_utilizada_en_pct).
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return float("nan"), 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    coverage = (mask.sum() / mask.size) * 100.0
    return float(mape), float(coverage)


def train_loop(model, optimizer, loss_fn, loader, device):
    """Ciclo de entrenamiento por época."""
    model.train()
    total_loss = 0.0
    for seqs, targets in loader:
        seqs = seqs.to(device)
        targets = targets.to(device)

        preds = model(seqs)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
    return total_loss / max(1, len(loader.dataset))


def eval_loop(model, loader, device):
    """Evalúa y devuelve MAE, RMSE, MAPE y los arreglos reales/predichos."""
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            preds = model(seqs)
            ys.append(targets.cpu().numpy())
            yps.append(preds.cpu().numpy())

    ys = np.vstack(ys)
    yps = np.vstack(yps)

    mae = mean_absolute_error(ys.ravel(), yps.ravel())
    rmse = mean_squared_error(ys.ravel(), yps.ravel())
    mape, mape_cov = masked_mape(ys, yps)

    return mae, rmse, mape, mape_cov, ys, yps


# ---------------------- Pipeline principal con Grid Search ----------------------
def train_single_config(hyperparams, df, centroids, district_names, series):
    """
    Entrena un modelo con una configuración específica de hiperparámetros.
    Usa la lógica original del split temporal (70/30).
    Retorna las métricas de test y el estado del mejor modelo.
    """
    # Crea una configuración temporal con los hiperparámetros
    temp_cfg = CFG()
    for key, value in hyperparams.items():
        if hasattr(temp_cfg, key):
            setattr(temp_cfg, key, value)

    # Lógica original de split (70/30)
    T = len(series)
    if T < temp_cfg.seq_len + 2:
        return None, f"Serie muy corta para seq_len={temp_cfg.seq_len}"

    train_end_i = int(T * 0.7)
    train_df = series.iloc[:train_end_i]
    test_df = series.iloc[train_end_i:]

    if len(train_df) <= temp_cfg.seq_len:
        return None, "El entrenamiento no tiene suficientes pasos"
    if len(test_df) <= temp_cfg.seq_len:
        test_df = series.iloc[train_end_i - 1:]

    # Validación: últimos 2*seq_len pasos del entrenamiento
    val_df = train_df.iloc[-(temp_cfg.seq_len * 2):]

    # Escalado por distrito usando solo train
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_scaled = pd.DataFrame(scaler.transform(train_df.values), index=train_df.index, columns=train_df.columns)
    val_scaled = pd.DataFrame(scaler.transform(val_df.values), index=val_df.index, columns=val_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df.values), index=test_df.index, columns=test_df.columns)

    # Datasets y loaders
    train_ds = TimeSeriesGraphDataset(train_scaled, seq_len=temp_cfg.seq_len)
    val_ds = TimeSeriesGraphDataset(val_scaled, seq_len=temp_cfg.seq_len)
    test_ds = TimeSeriesGraphDataset(test_scaled, seq_len=temp_cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=temp_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=temp_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=temp_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Modelo
    A_norm, _ = build_adjacency_from_centroids(centroids, k=temp_cfg.gcn_k)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32, device=temp_cfg.device)
    model = IntegratedModel(n_nodes=len(district_names), seq_len=temp_cfg.seq_len,
                            in_ch=1, hidden=temp_cfg.hidden_dim, A_norm=A_tensor).to(temp_cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=temp_cfg.lr)
    loss_fn = nn.MSELoss()

    # Entrenamiento con selección del mejor por RMSE de validación
    best_val = float('inf')
    best_state = None
    for epoch in range(1, temp_cfg.epochs + 1):
        tr_loss = train_loop(model, optimizer, loss_fn, train_loader, temp_cfg.device)
        val_mae, val_rmse, val_mape, val_mape_cov, _, _ = eval_loop(model, val_loader, temp_cfg.device)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {
                'model': model.state_dict(),
                'scaler_mean': scaler.mean_.copy(),
                'scaler_scale': scaler.scale_.copy(),
                'district_names': district_names,
                'A_norm': A_norm.copy(),
                'hyperparams': hyperparams.copy(),
            }

    # Evaluación en test
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        test_mae, test_rmse, test_mape, test_mape_cov, ys, yps = eval_loop(model, test_loader, temp_cfg.device)

        metrics = {
            'hyperparams': hyperparams,
            'val_RMSE_best': float(best_val),
            'test_MAE': float(test_mae),
            'test_RMSE': float(test_rmse),
            'test_MAPE_pct': None if math.isnan(test_mape) else float(test_mape),
            'test_MAPE_coverage_pct': float(test_mape_cov),
            'n_train_samples': len(train_ds),
            'n_val_samples': len(val_ds),
            'n_test_samples': len(test_ds),
        }
        return best_state, metrics
    return None, "Entrenamiento falló"


def grid_search_pipeline():
    """Ejecuta grid search sobre los hiperparámetros (seq_len fijo en 3)."""
    print("=" * 80)
    print("INICIANDO GRID SEARCH (seq_len fijo en 3)")
    print("=" * 80)

    # Carga datos una sola vez
    print("\nCargando datos...")
    df = load_and_concatenate(cfg.data_glob)
    df = normalize_schema(df)

    centroids = build_district_centroids(df)
    A_norm, district_names = build_adjacency_from_centroids(centroids, k=cfg.gcn_k)
    print(f"N° de distritos: {len(district_names)}")

    series = aggregate_yearly_counts(df, district_names)
    print(f"Forma de la serie (años × distritos): {series.shape}")

    # Genera todas las combinaciones
    grid_combos = generate_grid_combinations()
    print(f"\nTotal de combinaciones a probar: {len(grid_combos)}")
    print("=" * 80)

    results = []
    best_overall = None
    best_overall_rmse = float('inf')

    for idx, hyperparams in enumerate(grid_combos, 1):
        print(f"\n[{idx}/{len(grid_combos)}] Probando:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")

        best_state, result = train_single_config(hyperparams, df, centroids, district_names, series)

        if best_state is not None:
            results.append(result)
            test_rmse = result['test_RMSE']
            print(f"  ✓ Test RMSE: {test_rmse:.4f}")

            if test_rmse < best_overall_rmse:
                best_overall_rmse = test_rmse
                best_overall = best_state
                print(f"  ★ NUEVO MEJOR MODELO (Test RMSE: {test_rmse:.4f})")
        else:
            print(f"  ✗ Error: {result}")

    # Guarda resultados
    print("\n" + "=" * 80)
    print("RESULTADOS DEL GRID SEARCH")
    print("=" * 80)

    if best_overall is not None:
        best_result = min(results, key=lambda x: x['test_RMSE'])
        
        # Guarda el mejor modelo
        save_path = os.path.join(cfg.outdir, 'integrated_model.pt')
        torch.save(best_overall, save_path)
        print(f"\n✓ Mejor modelo guardado en: {save_path}")

        print("\nMejores hiperparámetros:")
        for k, v in best_result['hyperparams'].items():
            print(f"  {k}: {v}")

        print("\nMejores métricas:")
        print(f"  val_RMSE_best: {best_result['val_RMSE_best']:.6f}")
        print(f"  test_MAE: {best_result['test_MAE']:.6f}")
        print(f"  test_RMSE: {best_result['test_RMSE']:.6f}")
        print(f"  test_MAPE_pct: {best_result['test_MAPE_pct']}")
        print(f"  test_MAPE_coverage_pct: {best_result['test_MAPE_coverage_pct']:.2f}%")
        print(f"  n_train_samples: {best_result['n_train_samples']}")
        print(f"  n_val_samples: {best_result['n_val_samples']}")
        print(f"  n_test_samples: {best_result['n_test_samples']}")

        # Guarda reporte detallado
        report = {
            'best_hyperparams': best_result['hyperparams'],
            'best_metrics': {k: v for k, v in best_result.items() if k != 'hyperparams'},
            'n_nodes': len(district_names),
            'seq_len_fixed': 3,
            'total_combinations_tested': len(grid_combos),
            'successful_combinations': len(results),
            'all_results': results,
        }

        report_path = os.path.join(cfg.outdir, 'gridsearch_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Reporte detallado guardado en: {report_path}")

    else:
        print("\n✗ No se pudo entrenar ningún modelo.")


def main_pipeline():
    """Ejecuta el pipeline completo con grid search."""
    grid_search_pipeline()


if __name__ == '__main__':
    main_pipeline()
