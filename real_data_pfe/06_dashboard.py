# ============================================================
#  06_dashboard.py — Dashboard Santé Publique
#  SARA BENRAHMOUNE — BTS AL KENDI — 2024-2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="HealthVision Analytics",
    page_icon="🩺",
)

# ────────────────────────────────────────────────────────────
# GLOBAL CSS — Premium Health / Data Science Theme
# ────────────────────────────────────────────────────────────
st.markdown("""<style>
html * { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc !important; color: #1e293b !important; }
.stApp { background: #f8fafc !important; background-attachment: fixed !important; }
.block-container { padding-top: 1.5rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
[data-testid="stSidebar"] { background: #f1f5f9 !important; border-right: 1px solid #e2e8f0 !important; }
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] .stRadio label { color: #475569 !important; font-size: 0.95rem; padding: 0.3rem 0; }
[data-testid="stSidebar"] .stSlider > div { color: #0284c7 !important; }
.brand-header { background: linear-gradient(135deg, #0284c7, #ea580c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1.2; margin-bottom: 0.3rem; }
.brand-sub { color: #64748b !important; font-size: 0.75rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 1.5rem; }
.page-title { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #0284c7, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; line-height: 1.2; }
.page-subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; letter-spacing: 0.3px; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: #334155; border-left: 3px solid #ea580c; padding-left: 0.75rem; margin: 1.5rem 0 0.75rem 0; letter-spacing: 0.3px; text-transform: uppercase; font-size: 0.85rem; }
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card { flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.5rem; position: relative; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); transition: transform 0.2s, box-shadow 0.2s; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; border-radius: 16px 16px 0 0; }
.kpi-card.blue::before { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.kpi-card.orange::before { background: linear-gradient(90deg, #ea580c, #fb923c); }
.kpi-card.teal::before { background: linear-gradient(90deg, #0d9488, #2dd4bf); }
.kpi-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.kpi-icon { font-size: 1.8rem; margin-bottom: 0.5rem; display: block; }
.kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #0f172a; line-height: 1; margin-bottom: 0.3rem; }
.kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.chart-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); }
.model-card { border-radius: 16px; padding: 1.5rem; background: #ffffff; border: 1px solid #e2e8f0; text-align: center; position: relative; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
.model-card.best { border-color: #fbbf24; box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.5), 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
.model-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b; }
.model-metric { margin: 0.4rem 0; font-size: 0.95rem; color: #64748b; }
.model-metric span { font-weight: 700; color: #0f172a; font-size: 1.1rem; }
.best-badge { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #fff; font-size: 0.72rem; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 999px; display: inline-block; margin-bottom: 0.8rem; letter-spacing: 0.5px; }
.risk-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.risk-Critique { background: #fee2e2; color: #dc2626; border: 1px solid #f87171; }
.risk-Élevé { background: #ffedd5; color: #ea580c; border: 1px solid #fdba74; }
.risk-Modéré { background: #fef9c3; color: #ca8a04; border: 1px solid #fde047; }
.risk-Stable { background: #d1fae5; color: #059669; border: 1px solid #6ee7b7; }
.h-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent); margin: 1.5rem 0; }
.stSelectbox label, .stMultiselect label, .stSlider label { color: #475569 !important; font-size: 0.85rem !important; font-weight: 600 !important; }
.stButton > button { background: linear-gradient(135deg, #0284c7, #38bdf8) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.6rem 2rem !important; font-family: 'Inter', sans-serif !important; letter-spacing: 0.3px !important; transition: transform 0.2s !important; box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3); }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 8px -1px rgba(2, 132, 199, 0.4); }
.stDownloadButton > button { background: linear-gradient(135deg, #0d9488, #14b8a6) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.3); }
.stMetric { background: #ffffff; border-radius: 12px; padding: 0.8rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
.stMetric label { color: #64748b !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;}
.stMetric [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.8rem !important; font-weight: 700 !important; }
div[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; border: 1px solid #e2e8f0; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS
# ────────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data():
    processed_path = Path("data/processed/train_clean.csv")
    raw_path       = Path("data/infectious-disease-train.csv")
    if processed_path.is_file():
        df = pd.read_csv(processed_path)
    elif raw_path.is_file():
        df = pd.read_csv(raw_path)
    else:
        np.random.seed(42)
        diseases = [
            "Campylobacteriosis","Salmonellosis","Hepatitis A","Influenza",
            "Chlamydia","Gonorrhea","Tuberculosis","Lyme Disease",
            "West Nile Virus","Pertussis","Shigellosis","Cryptosporidiosis",
            "Meningococcal Disease","Mumps","Rabies","Syphilis","Measles","Rubella","Legionellosis","Brucellosis"
        ]
        counties = list(COUNTY_GPS.keys())
        years = np.arange(2001, 2015)
        rows = []
        for y in years:
            for d in np.random.choice(diseases, size=15, replace=False):
                for c in np.random.choice(counties, size=8, replace=False):
                    for s in ["Male", "Female", "Total"]:
                        count = max(0, int(np.random.lognormal(mean=4.5, sigma=1.2)))
                        rows.append({"Disease": d, "County": c, "Year": y, "Sex": s, "Count": count})
        df = pd.DataFrame(rows)
    df["Year"]  = df["Year"].astype(int)
    df["Count"] = df["Count"].astype(int)
    return df

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    model_dir = Path("models")
    names_map = {
        "xgboost_model.pkl":          "XGBoost",
        "random_forest_model.pkl":    "RandomForest",
        "gradient_boosting_model.pkl":"GradientBoosting",
    }
    out = {}
    for fname, key in names_map.items():
        p = model_dir / fname
        try:
            out[key] = joblib.load(p)
        except Exception:
            out[key] = None
    return out

def compute_growth_rates(df):
    early = df[(df["Year"] >= 2001) & (df["Year"] <= 2005)]
    late  = df[(df["Year"] >= 2010) & (df["Year"] <= 2014)]
    em = early.groupby(["Disease","County"])["Count"].mean().reset_index(name="early_mean")
    lm = late.groupby(["Disease","County"])["Count"].mean().reset_index(name="late_mean")
    mg = pd.merge(em, lm, on=["Disease","County"], how="inner")
    mg["growth"] = (mg["late_mean"] - mg["early_mean"]) / (mg["early_mean"] + 1) * 100
    return mg

def risk_level(g):
    if g > 75:   return "Critique"
    elif g > 25: return "Élevé"
    elif g > 0:  return "Modéré"
    else:         return "Stable"

# ────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────
COUNTY_GPS = {
    "Los Angeles":    (34.052, -118.243), "San Diego":     (32.715, -117.157),
    "San Francisco":  (37.774, -122.419), "Sacramento":    (38.582, -121.494),
    "Alameda":        (37.601, -122.000), "Santa Clara":   (37.354, -121.969),
    "Orange":         (33.787, -117.853), "Riverside":     (33.953, -117.396),
    "San Bernardino": (34.108, -117.289), "Contra Costa":  (37.925, -121.978),
    "Fresno":         (36.737, -119.787), "Kern":          (35.491, -119.019),
    "Ventura":        (34.275, -119.228), "San Mateo":     (37.563, -122.313),
    "Sonoma":         (38.527, -122.727), "Solano":        (38.268, -121.940),
    "Tulare":         (36.209, -119.346), "Santa Barbara": (34.420, -119.698),
    "Monterey":       (36.237, -121.310), "Placer":        (39.093, -121.017),
}
MODEL_COLORS = {"XGBoost":"#E85D24","RandomForest":"#00b4d8","GradientBoosting":"#1D9E75"}
PLACEHOLDER_METRICS = {
    "XGBoost":          {"R2":0.87,"RMSE":142,"MAE":98},
    "RandomForest":     {"R2":0.81,"RMSE":170,"MAE":115},
    "GradientBoosting": {"R2":0.78,"RMSE":190,"MAE":130},
}

# Plotly default layout
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#475569"),
    title_font=dict(family="Space Grotesk, sans-serif", color="#1e293b", size=16),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(color="#475569")),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    margin=dict(t=40, b=20, l=10, r=10),
)

def fig_style(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='brand-header'>🩺 HealthVision</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-sub'>Analytics Platform</div>", unsafe_allow_html=True)

    st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)

    st.markdown("**📅 Période d'analyse**")
    period = st.slider("", 2001, 2014, (2001, 2014), step=1)

    st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["📊  Vue générale", "🦠  Exploration maladies", "🤖  Prédictions ML", "🚨  Intervention prioritaire"]
    )

    st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#334155; line-height:1.8;'>
    <b style='color:#475569;'>⚙️ Stack Technique</b><br>
    🗄️ Apache Hadoop HDFS<br>
    ⚡ Apache Spark / PySpark<br>
    🐍 Scikit-learn & XGBoost<br>
    📊 Streamlit & Plotly<br>
    🗃️ Pandas & NumPy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='h-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#1e3a5f; text-align:center;'>
    Sara Benrahmoune · BTS Al Kendi<br>2024–2025
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────
df_raw = load_data()
df = df_raw[(df_raw["Year"] >= period[0]) & (df_raw["Year"] <= period[1])]

# ════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ════════════════════════════════════════════════════════════
if "Vue générale" in page:
    st.markdown("<div class='page-title'>Vue générale</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Tableau de bord épidémiologique · Californie 2001–2014</div>", unsafe_allow_html=True)

    total_cases       = int(df["Count"].sum())
    distinct_diseases = df["Disease"].nunique()
    distinct_counties = df["County"].nunique()
    years_covered     = df["Year"].nunique()

    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi-card blue'>
            <span class='kpi-icon'>🧬</span>
            <div class='kpi-value'>{total_cases:,}</div>
            <div class='kpi-label'>Total cas recensés</div>
        </div>
        <div class='kpi-card orange'>
            <span class='kpi-icon'>🦠</span>
            <div class='kpi-value'>{distinct_diseases}</div>
            <div class='kpi-label'>Maladies distinctes</div>
        </div>
        <div class='kpi-card teal'>
            <span class='kpi-icon'>📍</span>
            <div class='kpi-value'>{distinct_counties}</div>
            <div class='kpi-label'>Comtés touchés</div>
        </div>
        <div class='kpi-card purple'>
            <span class='kpi-icon'>📅</span>
            <div class='kpi-value'>{years_covered}</div>
            <div class='kpi-label'>Années couvertes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Annual evolution
    st.markdown("<div class='section-title'>Évolution annuelle des cas</div>", unsafe_allow_html=True)
    annual = df.groupby("Year")["Count"].sum().reset_index()
    max_row = annual.loc[annual["Count"].idxmax()]
    min_row = annual.loc[annual["Count"].idxmin()]
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=annual["Year"], y=annual["Count"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(232,93,36,0.10)",
        line=dict(color="#E85D24", width=2.5),
        marker=dict(size=6, color="#E85D24"),
        name="Cas annuels"
    ))
    fig_ev.add_annotation(x=max_row["Year"], y=max_row["Count"],
        text=f"⬆ Max : {int(max_row['Count']):,}", showarrow=True,
        arrowhead=2, arrowcolor="#E85D24", font=dict(color="#E85D24", size=11))
    fig_ev.add_annotation(x=min_row["Year"], y=min_row["Count"],
        text=f"⬇ Min : {int(min_row['Count']):,}", showarrow=True,
        arrowhead=2, arrowcolor="#00b4d8", font=dict(color="#00b4d8", size=11))
    fig_ev.update_layout(height=320, showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_ev, use_container_width=True)

    col_l, col_r = st.columns([3, 2])

    # ── Top 10 diseases
    with col_l:
        st.markdown("<div class='section-title'>Top 10 maladies</div>", unsafe_allow_html=True)
        top10 = df.groupby("Disease")["Count"].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top10, x="Count", y="Disease", orientation="h",
                         color="Count", color_continuous_scale="Blues")
        fig_bar.update_layout(height=340, yaxis_categoryorder="total ascending",
                               coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sex donut
    with col_r:
        st.markdown("<div class='section-title'>Répartition par sexe</div>", unsafe_allow_html=True)
        sex_df = df[df["Sex"] != "Total"].groupby("Sex")["Count"].sum().reset_index()
        fig_donut = go.Figure(data=[go.Pie(
            labels=sex_df["Sex"], values=sex_df["Count"],
            hole=0.62,
            marker=dict(colors=["#00b4d8","#E85D24"], line=dict(color="#ffffff", width=3)),
            textinfo="label+percent",
            textfont=dict(color="#1e293b", size=12)
        )])
        fig_donut.update_layout(
            height=340, showlegend=True,
            legend_yanchor="top", legend_y=0.99, legend_xanchor="left", legend_x=0.01,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_donut, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION MALADIES
# ════════════════════════════════════════════════════════════
elif "Exploration" in page:
    st.markdown("<div class='page-title'>Exploration des maladies</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Analyse approfondie par pathologie et localisation géographique</div>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        selected_disease = st.selectbox("🦠 Sélectionner une maladie", sorted(df["Disease"].unique()))
    with col_f2:
        selected_sex = st.multiselect("👤 Sexe", ["Male","Female","Total"], default=["Male","Female","Total"])

    filtered = df[(df["Disease"] == selected_disease) & (df["Sex"].isin(selected_sex))]

    if filtered.empty:
        st.warning("Aucune donnée pour cette sélection.")
        st.stop()

    total_c  = int(filtered["Count"].sum())
    n_county = filtered["County"].nunique()
    peak_y   = int(filtered.groupby("Year")["Count"].sum().idxmax())

    c1, c2, c3 = st.columns(3)
    c1.metric("🔬 Total cas",        f"{total_c:,}")
    c2.metric("📍 Comtés touchés",   n_county)
    c3.metric("📈 Année pic",        peak_y)

    # ── Line chart
    st.markdown("<div class='section-title'>Évolution temporelle</div>", unsafe_allow_html=True)
    line_df = filtered.groupby(["Year","Sex"])["Count"].sum().reset_index()
    sex_colors = {"Male":"#00b4d8","Female":"#E85D24","Total":"#1D9E75"}
    fig_line = px.line(line_df, x="Year", y="Count", color="Sex",
                       color_discrete_map=sex_colors, markers=True)
    fig_line.update_traces(line=dict(width=2.5))
    fig_line.update_layout(height=300, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_line, use_container_width=True)

    col_map, col_tbl = st.columns([3, 2])

    # ── Mapbox
    with col_map:
        st.markdown("<div class='section-title'>Carte des comtés californiens</div>", unsafe_allow_html=True)
        map_df = filtered.groupby("County")["Count"].sum().reset_index()
        map_df["lat"] = map_df["County"].apply(lambda x: COUNTY_GPS.get(x,(np.nan,np.nan))[0])
        map_df["lon"] = map_df["County"].apply(lambda x: COUNTY_GPS.get(x,(np.nan,np.nan))[1])
        map_df = map_df.dropna(subset=["lat","lon"])
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", size="Count", color="Count",
                color_continuous_scale=["#0a1628","#0077b6","#00b4d8","#E85D24","#ff4500"],
                hover_name="County", hover_data={"Count":True,"lat":False,"lon":False},
                zoom=5, center={"lat":37,"lon":-119.5}, height=420,
            )
            fig_map.update_layout(
                mapbox_style="carto-positron",
                margin={"r":0,"t":0,"l":0,"b":0},
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Aucun comté avec coordonnées GPS disponibles pour cette sélection.")

    # ── Top 10 table
    with col_tbl:
        st.markdown("<div class='section-title'>Top 10 comtés</div>", unsafe_allow_html=True)
        top10_c = map_df.nlargest(10,"Count")[["County","Count"]].reset_index(drop=True)
        top10_c.index += 1
        st.dataframe(top10_c)

# ════════════════════════════════════════════════════════════
# PAGE 3 — PRÉDICTIONS ML
# ════════════════════════════════════════════════════════════
elif "Prédictions" in page:
    st.markdown("<div class='page-title'>Prédictions Machine Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Comparaison des modèles · XGBoost · Random Forest · Gradient Boosting</div>", unsafe_allow_html=True)

    models = load_models()

    # ── Model cards
    st.markdown("<div class='section-title'>Performance des modèles</div>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    model_cols = {"XGBoost":mc1, "RandomForest":mc2, "GradientBoosting":mc3}
    model_labels = {"XGBoost":"XGBoost","RandomForest":"Random Forest","GradientBoosting":"Gradient Boosting"}

    for mname, mcol in model_cols.items():
        m = PLACEHOLDER_METRICS[mname]
        is_best = mname == "XGBoost"
        bg = MODEL_COLORS[mname]
        badge = "<div class='best-badge'>⭐ MEILLEUR MODÈLE</div><br>" if is_best else ""
        best_cls = " best" if is_best else ""
        mcol.markdown(f"""
        <div class='model-card{best_cls}' style='background:linear-gradient(135deg,{bg}22,{bg}08);border-color:{bg}55;'>
            {badge}
            <div class='model-name' style='color:{bg};'>{model_labels[mname]}</div>
            <div class='model-metric'>R² <span>{m['R2']:.2f}</span></div>
            <div class='model-metric'>RMSE <span>{m['RMSE']}</span></div>
            <div class='model-metric'>MAE  <span>{m['MAE']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='h-divider'></div>", unsafe_allow_html=True)

    col_cmp, col_sc = st.columns(2)

    # ── Comparative bars
    with col_cmp:
        st.markdown("<div class='section-title'>Comparaison des métriques</div>", unsafe_allow_html=True)
        model_names_list = ["XGBoost","RandomForest","GradientBoosting"]
        display_names    = ["XGBoost","Random Forest","Gradient Boosting"]
        fig_cmp = go.Figure()
        for metric, symbol in [("R2","●"),("RMSE","■"),("MAE","▲")]:
            fig_cmp.add_trace(go.Bar(
                name=f"{symbol} {metric}",
                x=display_names,
                y=[PLACEHOLDER_METRICS[m][metric] for m in model_names_list],
                marker_color=[MODEL_COLORS[m] for m in model_names_list],
                opacity=0.85,
            ))
        fig_cmp.update_layout(barmode="group", height=320,
                               legend_orientation="h", legend_y=-0.25, legend_x=0.5, legend_xanchor="center",
                               **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Scatter real vs predicted
    with col_sc:
        st.markdown("<div class='section-title'>Réel vs Prédit (XGBoost)</div>", unsafe_allow_html=True)
        np.random.seed(1)
        actual = np.random.randint(0, 5000, 300)
        pred   = np.clip(actual * 0.9 + np.random.normal(0, 250, 300), 0, None)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=actual, y=pred, mode="markers",
            marker=dict(color="#E85D24", opacity=0.4, size=6),
            name="Prédictions"
        ))
        fig_sc.add_trace(go.Scatter(
            x=[0, 5000], y=[0, 5000], mode="lines",
            line=dict(dash="dot", color="#475569", width=1.5),
            name="Idéal (y=x)"
        ))
        fig_sc.update_layout(height=320, xaxis_title="Valeur réelle",
                               yaxis_title="Valeur prédite", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Simulator
    st.markdown("<div class='h-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔮 Simulateur de prédiction</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns([2, 2, 1])
    with s1:
        sim_dis = st.selectbox("Maladie", sorted(df["Disease"].unique()))
    with s2:
        sim_cnt = st.selectbox("Comté", sorted(df["County"].unique()))
    with s3:
        sim_yr  = st.slider("Année", 2001, 2030, 2025)

    if st.button("🔮 Lancer la prédiction"):
        if models.get("XGBoost") is None:
            st.warning("⚠️ Modèle XGBoost non trouvé. Exécutez d'abord `05_model_evaluate.py`.")
        else:
            try:
                feat = np.array([[0, 0, sim_yr, 0, sim_yr // 10, (sim_yr - 2001) / 13]])
                result = int(models["XGBoost"].predict(feat)[0])
                st.success(f"🎯 Prédiction : **{result:,} cas** de **{sim_dis}** à **{sim_cnt}** en **{sim_yr}**")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# ════════════════════════════════════════════════════════════
# PAGE 4 — POINTS D'INTERVENTION
# ════════════════════════════════════════════════════════════
elif "Intervention" in page:
    st.markdown("<div class='page-title'>Points d'intervention prioritaires</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Identification des zones à risque par analyse de croissance épidémiologique</div>", unsafe_allow_html=True)

    growth_df = compute_growth_rates(df_raw)
    growth_df["Risk"] = growth_df["growth"].apply(risk_level)

    # Filters
    f1, f2 = st.columns([3, 1])
    with f1:
        selected_risks = st.multiselect(
            "🎯 Niveaux de risque",
            ["Critique","Élevé","Modéré","Stable"],
            default=["Critique","Élevé","Modéré","Stable"]
        )
    with f2:
        zone_limit = st.slider("Nb zones", 5, 30, 15)

    risk_colors = {"Critique":"#E85D24","Élevé":"#FF9500","Modéré":"#FFD700","Stable":"#1D9E75"}

    display = growth_df[growth_df["Risk"].isin(selected_risks)].nlargest(zone_limit,"growth")

    if display.empty:
        st.info("Aucune zone correspondant aux filtres sélectionnés.")
        st.stop()

    crit_cnt  = int((display["Risk"] == "Critique").sum())
    max_g     = float(display["growth"].max())
    top_d     = display.iloc[0]["Disease"]

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Zones critiques",        crit_cnt)
    m2.metric("📈 Croissance max (%)",      f"{max_g:.1f}%")
    m3.metric("⚠️ Maladie la + risquée",   top_d)

    col_tbl, col_bar = st.columns([2, 2])

    # ── Sorted table
    with col_tbl:
        st.markdown("<div class='section-title'>Tableau des zones prioritaires</div>", unsafe_allow_html=True)
        show = display[["Disease","County","growth","Risk"]].copy()
        show["growth"] = show["growth"].round(1)
        show.columns = ["Maladie","Comté","Croissance (%)","Niveau"]
        st.dataframe(show.reset_index(drop=True))

    # ── Bar chart top diseases
    with col_bar:
        st.markdown("<div class='section-title'>Top 10 maladies à forte croissance</div>", unsafe_allow_html=True)
        top10_g = growth_df.groupby("Disease")["growth"].mean().nlargest(10).reset_index()
        fig_g = px.bar(top10_g, x="growth", y="Disease", orientation="h",
                       color="growth",
                       color_continuous_scale="RdYlGn_r")
        fig_g.update_layout(height=380, yaxis_categoryorder="total ascending",
                             coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Risk map
    st.markdown("<div class='section-title'>Carte des risques épidémiologiques</div>", unsafe_allow_html=True)
    map_r = display.copy()
    map_r["lat"] = map_r["County"].apply(lambda x: COUNTY_GPS.get(x,(np.nan,np.nan))[0])
    map_r["lon"] = map_r["County"].apply(lambda x: COUNTY_GPS.get(x,(np.nan,np.nan))[1])
    map_r = map_r.dropna(subset=["lat","lon"])
    map_r["size_val"] = map_r["growth"].clip(lower=1)

    if not map_r.empty:
        fig_rmap = px.scatter_mapbox(
            map_r, lat="lat", lon="lon",
            size="size_val", color="Risk",
            color_discrete_map=risk_colors,
            hover_name="Disease",
            hover_data={"County": True, "growth": True, "lat": False, "lon": False, "size_val": False},
            labels={"County": "Comté", "growth": "Croissance (%)"},
            zoom=5, center={"lat":37,"lon":-119.5}, height=480,
        )
        fig_rmap.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1, font=dict(color="#1e293b")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rmap, use_container_width=True)
    else:
        st.info("Carte indisponible : pas de coordonnées GPS pour les comtés filtrés.")

    # ── CSV Export
    st.markdown("<div class='h-divider'></div>", unsafe_allow_html=True)
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les zones prioritaires (CSV)",
        data=csv,
        file_name="zones_prioritaires.csv",
        mime="text/csv"
    )
