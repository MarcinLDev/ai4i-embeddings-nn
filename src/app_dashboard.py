import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

import io, os
from pathlib import Path
import plotly.io as pio
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Ustawienia strony (tytuł, układ) – kolory/typografia wczyta config.toml
st.set_page_config(page_title="FactoryPulse – Utrzymanie predykcyjne", layout="wide")

from pathlib import Path

# --- LOGO na samej górze, osobno ---
logo_path = Path(r"../logo/BayLogicAI_logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=110)   # samodzielne logo u góry
else:
    st.warning("Nie znaleziono logo pod podaną ścieżką.")

@st.cache_data
def load_df():
    # czytamy plik wygenerowany przez build_dashboard_dataset.py
    try:
        df = pd.read_parquet("data/processed/ai4i2020_dashboard.parquet")
    except Exception:
        df = pd.read_csv("data/processed/ai4i2020_dashboard.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df = load_df()

# ==================== SIDEBAR / FILTRY ====================
st.sidebar.header("Filtry")
tmin, tmax = df["Timestamp"].min(), df["Timestamp"].max()
zakres_czasu = st.sidebar.slider(
    "Zakres czasu",
    min_value=tmin.to_pydatetime(),
    max_value=tmax.to_pydatetime(),
    value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
)

typy = st.sidebar.multiselect(
    "Typ produktu",
    sorted(df["Type"].dropna().unique()),
    default=list(sorted(df["Type"].dropna().unique()))
)

typy_awarii = st.sidebar.multiselect(
    "Typ awarii",
    sorted(df["Failure Type"].dropna().unique()),
    default=list(sorted(df["Failure Type"].dropna().unique()))
)

zrodlo_modelu = st.sidebar.selectbox(
    "Pokaż prawdopodobieństwo z modelu",
    options=["proba_nn", "proba_xgb"],
    format_func=lambda x: "Sieć NN" if x == "proba_nn" else "XGBoost"
)

okno_ma = st.sidebar.select_slider(
    "Wygładzanie (średnia krocząca)",
    options=["brak", "15min", "60min"],
    value="60min"
)

mask = (
    df["Timestamp"].between(*zakres_czasu)
    & df["Type"].isin(typy)
    & df["Failure Type"].isin(typy_awarii)
)
d = df.loc[mask].copy()

# ==================== KPI / KARTY ====================
st.title("Monitoring anomalii i awarii maszyny produkcyjnej")

col1, col2, col3, col4 = st.columns(4)
liczba_awarii = int(d["Anomaly"].sum())
czestosc = (liczba_awarii / max(1, len(d))) * 100
srednie_zuzycie_przy_awarii = d.loc[d["Anomaly"] == 1, "Tool wear [min]"].mean()
srednie_proba = d[zrodlo_modelu].mean()

with col1:
    st.metric("Liczba wykrytych awarii", value=f"{liczba_awarii}")
with col2:
    st.metric("Odsetek awarii", value=f"{czestosc:.2f}%")
with col3:
    st.metric(
        "Średnie zużycie narzędzia przy awarii",
        value=("—" if not np.isfinite(srednie_zuzycie_przy_awarii) else f"{srednie_zuzycie_przy_awarii:.1f} min")
    )
with col4:
    st.metric(
        f"Śr. prawdopodobieństwo ({'NN' if zrodlo_modelu=='proba_nn' else 'XGB'})",
        value=("—" if not np.isfinite(srednie_proba) else f"{srednie_proba:.3f}")
    )

st.markdown("---")

# ==================== LINIA: CZAS (czytelne serie bez słupków) ====================
st.subheader("Awarie / ryzyko w czasie")

# wykryj, czy są realne predykcje modeli (nie NaN)
ma_nn = ("proba_nn" in d.columns) and d["proba_nn"].notna().any()
ma_xgb = ("proba_xgb" in d.columns) and d["proba_xgb"].notna().any()
ma_model = ma_nn or ma_xgb

opcja = st.radio(
    "Wybierz serię:",
    options=(
        ["Udział awarii – linia wygładzona (obszar)",
         "Skumulowana liczba awarii – linia"]
        + (["Średnie ryzyko modelu w czasie (linia)"] if ma_model else [])
    ),
    horizontal=True
)

# wybór interwału i wygładzenia
colA, colB = st.columns(2)
with colA:
    interwal = st.selectbox("Agregacja czasowa", ["5min", "15min", "30min", "60min"], index=1)
with colB:
    okno = st.slider("Okno wygładzenia (liczba kubełków)", min_value=3, max_value=48, value=8, step=1)

def ewma_seria(s: pd.Series, window: int) -> pd.Series:
    alpha = 2 / (window + 1)
    return s.ewm(alpha=alpha, adjust=False).mean()

if opcja.startswith("Udział awarii"):
    # 1) resampling do udziału awarii w interwale
    ser = (
        d.set_index("Timestamp")
         .resample(interwal)["Anomaly"]
         .mean()
         .rename("udzial")
    )
    # 2) wygładzenie EWMA (ładna, gładka linia)
    sm = ewma_seria(ser, okno).rename("udzial_ewma")
    df_line = pd.concat([ser, sm], axis=1).reset_index()

    # obszar pod linią wygładzoną (bez słupków)
    fig_line = px.area(
        df_line, x="Timestamp", y="udzial_ewma",
        title=f"Udział awarii (agregacja: {interwal}, EWMA okno={okno})",
        labels={"udzial_ewma": "Udział awarii"}
    )
    # opcjonalnie cienka, półprzezroczysta linia surowa (możesz usunąć)
    fig_line.add_scatter(
        x=df_line["Timestamp"], y=df_line["udzial"],
        mode="lines", name="Udział surowy", opacity=0.25
    )
    st.plotly_chart(fig_line, use_container_width=True)

elif opcja.startswith("Skumulowana liczba"):
    cum = d.sort_values("Timestamp").copy()
    cum["Skumulowane awarie"] = cum["Anomaly"].cumsum()
    fig_cum = px.line(
        cum, x="Timestamp", y="Skumulowane awarie",
        title="Skumulowana liczba awarii (linia)",
        labels={"Skumulowane awarie": "Liczba [narastająco]"}
    )
    st.plotly_chart(fig_cum, use_container_width=True)

else:
    # Średnie ryzyko modelu w czasie (jeśli dostępne)
    # wybierz źródło: preferuj NN, jeśli brak to XGB
    kol = "proba_nn" if ma_nn else "proba_xgb"
    nazwa = "NN" if ma_nn else "XGB"
    risk = (
        d.set_index("Timestamp")
         .resample(interwal)[kol]
         .mean()
         .rename("ryzyko")
    )
    risk_sm = ewma_seria(risk, okno).rename("ryzyko_ewma")
    df_risk = pd.concat([risk, risk_sm], axis=1).reset_index()

    fig_risk = px.line(
        df_risk, x="Timestamp", y="ryzyko_ewma",
        title=f"Średnie ryzyko modelu w czasie ({nazwa}; {interwal}, EWMA okno={okno})",
        labels={"ryzyko_ewma": "Prawdopodobieństwo awarii"}
    )
    # delikatnie pokaż serię surową
    fig_risk.add_scatter(
        x=df_risk["Timestamp"], y=df_risk["ryzyko"],
        mode="lines", name="Ryzyko surowe", opacity=0.25
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ==================== SCATTER: PARAMETRY ====================
st.subheader("Parametry operacyjne vs. anomalia")
fig_sc = px.scatter(
    d,
    x="Torque [Nm]",
    y="Rotational speed [rpm]",
    color="Failure Type",
    symbol=d["Anomaly"].map({0: "Normalne", 1: "Anomalia"}),
    hover_data=["Timestamp", "Type", zrodlo_modelu],
    title="Moment obrotowy vs prędkość obrotowa"
)
fig_sc.update_traces(marker=dict(opacity=0.85))
st.plotly_chart(fig_sc, use_container_width=True)

# ==================== BAR: ROZKŁAD AWARII ====================
st.subheader("Rozkład awarii wg typu")
bar = d["Failure Type"].value_counts().rename_axis("Typ awarii").reset_index(name="Liczba")
fig_bar = px.bar(bar, x="Typ awarii", y="Liczba", title="Rozkład typów awarii")
st.plotly_chart(fig_bar, use_container_width=True)

# ==================== HEATMAP: KORELACJE ====================
# ==================== MAPA RYZYKA (zamiast macierzy korelacji) ====================
st.subheader("Mapa ryzyka pracy maszyny (jakie reżimy są niebezpieczne)")

# 1) Empiryczna mapa ryzyka: P(awaria) w siatce [Torque] × [Speed]
bins_torque = st.slider("Liczba podziałów momentu [Nm]", 8, 40, 18)
bins_speed  = st.slider("Liczba podziałów prędkości [rpm]", 8, 40, 18)

torque_bins = pd.cut(d["Torque [Nm]"], bins=bins_torque)
speed_bins  = pd.cut(d["Rotational speed [rpm]"], bins=bins_speed)

risk_grid = (
    d.assign(_torque_bin=torque_bins, _speed_bin=speed_bins)
      .groupby(["_torque_bin", "_speed_bin"], as_index=False)["Anomaly"]
      .mean()
      .rename(columns={"Anomaly": "Ryzyko"})
)

# centroidy przedziałów – lepszy tooltip/pozycjonowanie
risk_grid["Torque_c"] = risk_grid["_torque_bin"].apply(lambda x: x.mid)
risk_grid["Speed_c"]  = risk_grid["_speed_bin"].apply(lambda x: x.mid)

fig_riskmap = px.density_heatmap(
    risk_grid, x="Torque_c", y="Speed_c", z="Ryzyko",
    nbinsx=bins_torque, nbinsy=bins_speed,
    histfunc="avg",
    title="Prawdopodobieństwo awarii w przestrzeni pracy (Torque × Speed)",
    labels={"Torque_c":"Moment [Nm]", "Speed_c":"Prędkość [rpm]", "Ryzyko":"P(awaria)"},
    color_continuous_scale="RdYlGn_r", # czerwone = gorzej
)
fig_riskmap.update_yaxes(autorange="reversed")  # intuicyjny układ „mapy”
fig_riskmap.update_layout(coloraxis_colorbar=dict(title="Ryzyko"))

st.plotly_chart(fig_riskmap, use_container_width=True)

# 2) Top „gorące strefy” – gdzie ryzyko jest najwyższe (dla prezentacji zarządu)
topN = 8
hotspots = (risk_grid.sort_values("Ryzyko", ascending=False)
            .head(topN)
            .assign(Torque_bin=lambda df: df["_torque_bin"].astype(str),
                    Speed_bin=lambda df: df["_speed_bin"].astype(str))
            [["Torque_bin","Speed_bin","Ryzyko"]])

st.markdown("**Najbardziej ryzykowne strefy pracy (TOP)**")
st.dataframe(hotspots.style.format({"Ryzyko":"{:.1%}"}), use_container_width=True)

# 3) „Co-jeśli” – szybka symulacja ryzyka dla zadanych parametrów
st.subheader("Co-jeśli: jak zmiana parametrów wpływa na ryzyko?")
c1, c2, c3 = st.columns(3)
with c1:
    what_torque = st.slider("Moment [Nm] (what-if)", float(d["Torque [Nm]"].min()), float(d["Torque [Nm]"].max()), float(d["Torque [Nm]"].median()))
with c2:
    what_speed  = st.slider("Prędkość [rpm] (what-if)", float(d["Rotational speed [rpm]"].min()), float(d["Rotational speed [rpm]"].max()), float(d["Rotational speed [rpm]"].median()))
with c3:
    what_type = st.selectbox("Typ produktu (opcjonalnie)", ["(wszystkie)"] + list(d["Type"].dropna().unique()))

# jeśli masz predykcje modelu (NN/XGB), użyjemy ich; w przeciwnym razie weźmiemy wartości z najbliższego kubełka (empiryczne)
# --- FUNKCJA POMOCNICZA: czy wartość wpada do przedziału (lewo/prawo domknięte) ---
def _in_interval(iv, val):
    # pandas.Interval: iv.closed ∈ {'right','left','both','neither'}
    if iv.closed == 'right':
        return (val > iv.left) and (val <= iv.right)
    elif iv.closed == 'left':
        return (val >= iv.left) and (val < iv.right)
    elif iv.closed == 'both':
        return (val >= iv.left) and (val <= iv.right)
    else:  # 'neither'
        return (val > iv.left) and (val < iv.right)

# --- ESTYMACJA RYZYKA (model jeśli jest, inaczej empiryczny kubełek) ---
def estimate_risk(torque, speed, typ=None):
    # 1) Jeśli są predykcje modeli – lokalna średnia w okolicy (preferuj NN)
    def _local_mean_prob(colname):
        sub = d if (typ in [None, "(wszystkie)"]) else d[d["Type"] == typ]
        if colname not in sub.columns or not sub[colname].notna().any():
            return None
        # okno adaptacyjne (percentyle) – „sąsiedztwo” punktu roboczego
        t_w = max(sub["Torque [Nm]"].quantile(0.025), 1e-6)
        s_w = max(sub["Rotational speed [rpm]"].quantile(0.025), 1e-6)
        box = sub[
            sub["Torque [Nm]"].between(torque - t_w, torque + t_w)
            & sub["Rotational speed [rpm]"].between(speed - s_w, speed + s_w)
        ]
        if len(box) >= 20 and box[colname].notna().any():
            return float(box[colname].mean())
        return None

    for col in ["proba_nn", "proba_xgb"]:
        val = _local_mean_prob(col)
        if val is not None:
            return val

    # 2) Fallback: empiryczne P(awaria) w najbliższym kubełku (BEZ .iloc!)
    m_t = risk_grid["_torque_bin"].apply(lambda iv: _in_interval(iv, torque))
    m_s = risk_grid["_speed_bin"].apply(lambda iv: _in_interval(iv, speed))
    row = risk_grid.loc[m_t & m_s]   # <- .loc z maską bool (zamiast .iloc)

    if not row.empty:
        return float(row["Ryzyko"].iloc[0])

    # Gdy nic nie trafiło (np. wartości skrajne) – zwróć średnie globalne
    return float(d["Anomaly"].mean())

# wynik „co‑jeśli”
risk_est = estimate_risk(what_torque, what_speed, what_type)
st.metric("Szacowane ryzyko awarii", f"{risk_est*100:.1f}%")

# 4) Wersja „dla prezesa”: krótki komentarz (auto-insight)
# znajdź granice „czerwonych stref”
red_thresh = 0.15  # np. 15%+ uznajemy za strefę czerw.
red_share = (risk_grid["Ryzyko"] >= red_thresh).mean()
st.caption(
    f"Interpretacja: przy obecnych danych ~{red_share*100:.1f}% przestrzeni pracy (Torque×Speed) to strefy wysokiego ryzyka (≥{int(red_thresh*100)}%). "
    "Unikaj długiej pracy w tych okolicach lub obniż moment/prędkość – ryzyko spada."
)

# ==================== EKSPORT ====================
# anom = d[d["Anomaly"] == 1]
# st.download_button(
#     "Eksportuj przefiltrowane anomalie (CSV)",
#     anom.to_csv(index=False).encode("utf-8"),
#     file_name="anomalie_przefiltrowane.csv",
#     mime="text/csv"
# )

# 1) Rejestracja fontu z polskimi znakami
def _register_font():
    candidates = [
        Path("assets/fonts/DejaVuSans.ttf"),
        Path("assets/DejaVuSans.ttf"),
        Path("DejaVuSans.ttf"),
        Path(r"C:\Windows\Fonts\DejaVuSans.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                pdfmetrics.registerFont(TTFont("PLFONT", str(p)))
                return "PLFONT"
            except Exception:
                pass
    # fallback – brak PL znaków (ostrzegamy w UI)
    return None

FONT_NAME = _register_font()

def _set_font(c, size=12, bold=False):
    if FONT_NAME:
        c.setFont(FONT_NAME, size)
    else:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)

# 2) Render plotly → PNG (kolor)
def fig_to_png_bytes(fig):
    # jeśli kaleido nie działa, ta funkcja rzuci wyjątek
    return pio.to_image(fig, format="png", scale=2, engine="kaleido")

def draw_title(c, text, y):
    _set_font(c, 16, bold=True)
    c.drawString(2*cm, y, text)
    return y - 1.0*cm

def draw_kpis(c, y, kpis):
    _set_font(c, 12)
    lines = [
        f"Liczba wykrytych awarii: {kpis['total_fail']}",
        f"Odsetek awarii: {kpis['failure_rate']:.2f}%",
        "Średnie zużycie narzędzia przy awarii: " + (
            "—" if not np.isfinite(kpis['avg_wear']) else f"{kpis['avg_wear']:.1f} min"
        ),
    ]
    for line in lines:
        c.drawString(2*cm, y, line)
        y -= 0.75*cm
    return y - 0.25*cm

def draw_plot(c, fig, y, title=None):
    page_w, page_h = A4
    if title:
        _set_font(c, 13, bold=True)
        c.drawString(2*cm, y, title); y -= 0.8*cm

    # wygeneruj PNG
    png_bytes = fig_to_png_bytes(fig)
    img = ImageReader(io.BytesIO(png_bytes))

    iw, ih = img.getSize()
    max_w = page_w - 4*cm
    scale = max_w / iw
    w = max_w
    h = ih * scale

    # nowa strona jeśli nie mieści się w pionie
    if y - h < 2*cm:
        c.showPage(); y = page_h - 2*cm

    c.drawImage(img, 2*cm, y - h, width=w, height=h)
    return y - h - 0.6*cm

def build_pdf_bytes(figures, kpis, logo_path=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    y = page_h - 2*cm

    # nagłówek z logo
    if logo_path and Path(logo_path).exists():
        try:
            logo = ImageReader(str(logo_path))
            c.drawImage(logo, 2*cm, y-1.2*cm, width=3.2*cm, height=1.2*cm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    _set_font(c, 18, bold=True)
    c.drawString(6*cm, y-0.3*cm, "Raport – FactoryPulse (AI4I 2020)")
    y -= 1.8*cm

    # sekcja KPI
    y = draw_title(c, "KPI (bieżące filtry)", y)
    y = draw_kpis(c, y, kpis)

    # wykresy
    for title, fig in figures:
        try:
            y = draw_plot(c, fig, y, title=title)
        except Exception as e:
            # jeśli kaleido padnie – pokaż info w PDF zamiast białej dziury
            _set_font(c, 11)
            c.drawString(2*cm, y, f"[Błąd renderowania wykresu: {title}] {e}")
            y -= 1.0*cm

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue(); buf.close()
    return pdf_bytes

# ====== PRZYCISK W UI ======
if st.button("📄 Pobierz raport PDF"):
    if not FONT_NAME:
        st.info("Uwaga: nie znaleziono fontu PL. Aby mieć polskie znaki w PDF, dodaj assets/fonts/DejaVuSans.ttf lub wskaż Arial.")
    try:
        kpis_for_pdf = {
            "total_fail": liczba_awarii,
            "failure_rate": czestosc,
            "avg_wear": srednie_zuzycie_przy_awarii,
        }
        figs_for_pdf = [
            ("Awarie / ryzyko w czasie", fig_line),
            ("Parametry operacyjne vs anomalia", fig_sc),
            ("Rozkład awarii wg typu", fig_bar),
            ("Korelacje: parametry vs anomalia", fig_hm),
        ]
        pdf_bytes = build_pdf_bytes(
            figs_for_pdf, kpis_for_pdf,
            logo_path=r"C:\Users\marci\airflow\logo\BayLogicAI_logo.png"
        )
        st.download_button(
            "⬇️ Pobierz wygenerowany raport (PDF)",
            data=pdf_bytes,
            file_name="raport_factorypulse.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Nie udało się wygenerować PDF (sprawdź instalację 'kaleido' i 'reportlab'). Szczegóły: {e}")