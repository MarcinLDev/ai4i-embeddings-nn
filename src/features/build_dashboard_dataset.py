# -*- coding: utf-8 -*-
"""
Tworzy zestaw 'dashboardowy' z predykcjami modeli + KPI w czasie.
Uruchom:
    python -m src.features.build_dashboard_dataset
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ===== (opcjonalnie) modele =====
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import torch
    from torch import nn
except Exception:
    torch = None

# ===== ŚCIEŻKI PROJEKTU =====
ROOT = Path(".")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ART = ROOT / "models" / "artifacts"
CKPT = ROOT / "models" / "checkpoints"

RAW_FILE = RAW / "ai4i2020.csv"
OUT_PARQUET = PROC / "ai4i2020_dashboard.parquet"
OUT_CSV = PROC / "ai4i2020_dashboard.csv"


# ===== NARZĘDZIA =====
def ensure_dirs():
    PROC.mkdir(parents=True, exist_ok=True)


def safe_joblib(path: Path):
    """Bezpieczne wczytanie joblib/pickle; zwraca None gdy brak pliku."""
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


def add_timestamp(df: pd.DataFrame, start="2024-01-01 08:00:00", freq="1min"):
    """Dodaje syntetyczny Timestamp co 'freq' od 'start'."""
    df = df.copy()
    df["Timestamp"] = pd.date_range(start, periods=len(df), freq=freq)
    return df


def _derive_failure_type_from_flags(df: pd.DataFrame) -> pd.Series:
    """Buduje 'Failure Type' z flag TWF/HDF/PWF/OSF/RNF (lub 'No Failure')."""
    name_map = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failures",
    }
    flag_cols = [c for c in name_map.keys() if c in df.columns]
    if not flag_cols:
        return pd.Series(["No Failure"] * len(df), index=df.index)

    # w AI4I co najwyżej jedna flaga = 1
    idxmax = df[flag_cols].idxmax(axis=1)
    has_any = df[flag_cols].sum(axis=1) > 0
    ft = idxmax.map(name_map)
    return ft.where(has_any, "No Failure")


def basic_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica kolumny labelowe:
      - Anomaly (0/1)
      - Failure Type (tekst)
    Obsługuje: Target / Machine failure / Failure Type / flagi TWF,HDF,PWF,OSF,RNF.
    """
    df = df.copy()

    # === Anomaly ===
    if "Target" in df.columns:
        df["Anomaly"] = df["Target"].astype(int)
    elif "Machine failure" in df.columns:
        df["Anomaly"] = df["Machine failure"].astype(int)
    elif "Failure Type" in df.columns:
        df["Anomaly"] = (df["Failure Type"].astype(str) != "No Failure").astype(int)
    elif any(c in df.columns for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]):
        flags = df[[c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]]
        df["Anomaly"] = (flags.sum(axis=1) > 0).astype(int)
    else:
        raise ValueError(
            "Nie znaleziono kolumny z etykietą awarii (Target / Machine failure / Failure Type / flagi)."
        )

    # === Failure Type ===
    if "Failure Type" not in df.columns:
        df["Failure Type"] = _derive_failure_type_from_flags(df)

    return df


def infer_feature_order():
    """Jeśli istnieje X_train.csv, użyjemy jego kolejności kolumn (najbezpieczniej względem treningu)."""
    x_train = PROC / "X_train.csv"
    if x_train.exists():
        cols = pd.read_csv(x_train, nrows=1).columns.tolist()
        return cols
    return None


def prepare_features_from_raw(df: pd.DataFrame):
    """
    Spójny preprocessing z treningiem:
      - kolejność kolumn jak w X_train.csv (jeśli istnieje),
      - encodery/skalery z models/artifacts (jeśli istnieją),
      - bezpieczne fallbacki gdy artefaktów brak.
    """
    train_cols = infer_feature_order()

    # domyślny minimalny zestaw (gdy nie używamy X_train.csv)
    num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    cat_cols = ["Type"]  # "Product ID" zwykle pomijamy albo enkodujemy osobno

    if train_cols is not None:
        base = {}
        for c in train_cols:
            if c in df.columns:
                base[c] = df[c]
            elif c == "Type_idx" and "Type" in df.columns:
                base[c] = df["Type"]
            else:
                # brakujące kolumny (np. one-hot z treningu) – uzupełniamy zerami
                base[c] = 0
        X = pd.DataFrame(base)
    else:
        X = df[num_cols + cat_cols].copy()

    # ===== encodery/skalery z artifacts =====
    type_encoder = safe_joblib(ART / "type_encoder.joblib")
    product_encoder = safe_joblib(ART / "product_encoder.joblib")
    scaler = safe_joblib(ART / "scaler.joblib")

    # --- kodowanie 'Type' ---
    if "Type" in X.columns:
        if type_encoder is not None:
            X["Type"] = type_encoder.transform(X["Type"])
        else:
            X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2}).astype("Int64")

    # --- (opcjonalnie) Product ID ---
    if "Product ID" in X.columns:
        if product_encoder is not None:
            X["Product ID"] = product_encoder.transform(X["Product ID"])
        # else: zostawiamy jak jest / ewentualnie można usunąć kolumnę

    # --- standaryzacja liczb ---
    if scaler is not None:
        num_mask = X.columns.difference(["Type", "Product ID"])
        try:
            X[num_mask] = scaler.transform(X[num_mask])
        except Exception:
            pass
    else:
        num_mask = X.select_dtypes(include=[np.number]).columns
        X[num_mask] = (X[num_mask] - X[num_mask].mean()) / (X[num_mask].std() + 1e-9)

    return X


# ===== MODELE =====
def load_xgb():
    path = CKPT / "xgb_model.json"
    if xgb is None or not path.exists():
        warnings.warn("XGBoost niedostępny – pomijam predykcje XGB.")
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


class SimpleNN(nn.Module):
    """Placeholder na wypadek braku implementacji; podmień na własną klasę jeśli trzeba."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def load_nn(in_dim):
    ckpt = CKPT / "nn_model.pt"
    if torch is None or not ckpt.exists():
        warnings.warn("NN niedostępny – pomijam predykcje NN.")
        return None
    model = SimpleNN(in_dim)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def add_time_kpis(df: pd.DataFrame):
    """Dodaje rolling KPI: odsetek anomalii w oknach 15m/60m (na podstawie pełnej serii minutowej)."""
    s = df.sort_values("Timestamp").set_index("Timestamp")["Anomaly"]
    # rolling na indeksie czasowym
    df["AnomalyRate_15m"] = (
        s.rolling("15min").mean().reindex(df["Timestamp"].values, method="nearest").values
    )
    df["AnomalyRate_60m"] = (
        s.rolling("60min").mean().reindex(df["Timestamp"].values, method="nearest").values
    )
    return df


# ===== GŁÓWNY PIPELINE =====
def main():
    ensure_dirs()
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {RAW_FILE}")

    # 1) Wczytanie surowych danych
    raw = pd.read_csv(RAW_FILE)

    # 2) Czas + etykiety
    raw = add_timestamp(raw)        # Timestamp co 1 minutę
    raw = basic_cols(raw)           # Anomaly + Failure Type

    # 3) Przygotowanie cech spójnie z treningiem
    X = prepare_features_from_raw(raw)

    # 4) Predykcje modeli (jeśli są checkpointy)
    xgb_model = load_xgb()
    if xgb_model is not None:
        raw["proba_xgb"] = xgb_model.predict_proba(X)[:, 1]
        raw["pred_xgb"] = (raw["proba_xgb"] >= 0.5).astype(int)
    else:
        raw["proba_xgb"] = np.nan
        raw["pred_xgb"] = np.nan

    nn_model = load_nn(X.shape[1])
    if nn_model is not None:
        with torch.no_grad():
            p = nn_model(torch.tensor(X.values, dtype=torch.float32)).squeeze(1).numpy()
        raw["proba_nn"] = p
        raw["pred_nn"] = (p >= 0.5).astype(int)
    else:
        raw["proba_nn"] = np.nan
        raw["pred_nn"] = np.nan

    # 5) KPI po czasie
    raw = add_time_kpis(raw)

    # 6) Zapis (parquet + csv)
    raw.to_parquet(OUT_PARQUET, index=False)
    raw.to_csv(OUT_CSV, index=False)

    print("✅ Zapisano pliki dashboardowe:")
    print(f"   • {OUT_PARQUET}")
    print(f"   • {OUT_CSV}")


if __name__ == "__main__":
    main()
