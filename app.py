import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ----------------------------
# CONFIGURATION & DATA
# ----------------------------
st.set_page_config(page_title="Green IT ROI Premium", layout="wide")

BRAND_MODIFIERS = {
    "Apple": {"maint": 1.45, "lifetime": 1.25, "carbon": 0.85, "energy": 0.90},
    "Cisco": {"maint": 1.10, "lifetime": 1.50, "carbon": 0.90, "energy": 1.30},
    "Dell": {"maint": 0.90, "lifetime": 1.00, "carbon": 1.00, "energy": 1.00},
    "Lenovo": {"maint": 0.95, "lifetime": 1.05, "carbon": 1.00, "energy": 1.00},
    "HP": {"maint": 1.10, "lifetime": 0.95, "carbon": 1.05, "energy": 1.00},
    "Samsung": {"maint": 1.20, "lifetime": 1.00, "carbon": 1.10, "energy": 1.05},
    "Autre": {"maint": 1.00, "lifetime": 0.90, "carbon": 1.20, "energy": 1.10},
}

DEVICE_TYPES = ["Laptop", "Tablet", "Smartphone", "Switch / Router", "Room Screens"]

# ----------------------------
# STYLE
# ----------------------------
st.markdown(
    """
    <style>
    .roi-box { background-color: #f8fafc; padding: 30px; border-radius: 15px; border-left: 10px solid #10b981; text-align: center; margin-bottom: 25px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .explanation-note { background-color: #f1f5f9; padding: 15px; border-radius: 8px; border-left: 5px solid #64748b; font-style: italic; font-size: 0.9em; margin-top: -20px; margin-bottom: 20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# HELPERS
# ----------------------------
def clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def score_from_ratio(alt_value, ref_value):
    """
    Score stable:
    - 50 = neutre (alt == ref)
    - >50 si alt est meilleur (plus petit)
    - <50 si alt est pire (plus grand)
    """
    ratio = alt_value / max(ref_value, 1e-9)
    s = 50 + 50 * (1 - ratio)
    return clip(s, 0, 100)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Calculation Parameters")
material_type = st.sidebar.selectbox("Device Type", DEVICE_TYPES)

# Restriction Smartphone : Uniquement Apple
if material_type == "Smartphone":
    brand_list = ["Apple"]
    st.sidebar.info("Note: Smartphones limités à Apple (sécurité / parc).")
else:
    brand_list = list(BRAND_MODIFIERS.keys())

brand = st.sidebar.selectbox("Brand", brand_list)

price = st.sidebar.number_input("New Purchase Price (€)", min_value=0.0, value=1200.0)
carbon_prod = st.sidebar.number_input("CO₂ Blueprint (Prod/Transport) in kg", min_value=0.0, value=300.0)
base_lifetime = st.sidebar.number_input("Theoretical constructor duration (years)", min_value=1.0, value=5.0)
energy_annual = st.sidebar.number_input("Yearly Electrical Consumption (kWh/year)", min_value=0.0, value=150.0)
maint_base = st.sidebar.number_input("Maintenance Budget Allowed (€)", min_value=0.0, value=500.0)

# ✅ volumétrie par type (persistante)
st.sidebar.subheader("🏢 Fleet volumetry (by device type)")
if "FLEET_COUNTS" not in st.session_state:
    st.session_state["FLEET_COUNTS"] = {t: 0 for t in DEVICE_TYPES}
    st.session_state["FLEET_COUNTS"]["Laptop"] = 100  # valeur de départ

for t in DEVICE_TYPES:
    st.session_state["FLEET_COUNTS"][t] = st.sidebar.number_input(
        f"{t} count",
        min_value=0,
        value=int(st.session_state["FLEET_COUNTS"][t]),
        step=10,
        key=f"fleet_count_{t}",
    )

fleet_size_current_type = int(st.session_state["FLEET_COUNTS"][material_type])
fleet_size_total = int(sum(st.session_state["FLEET_COUNTS"].values()))
st.sidebar.caption(f"Current type volume: {fleet_size_current_type} | Total fleet: {fleet_size_total}")

# ✅ paramètres France/LVMH (modifiables)
st.sidebar.subheader("⚡ France / LVMH assumptions")
ELEC_PRICE = st.sidebar.number_input("Electricity price (€/kWh)", min_value=0.0, value=0.22, step=0.01)
CO2_FACTOR = st.sidebar.number_input("Grid CO₂ factor (kgCO₂e/kWh)", min_value=0.0, value=0.0217, step=0.001)

# ----------------------------
# LOGIQUE D'ÉLASTICITÉ (MAINTENANCE -> DURÉE DE VIE) + GARDE-FOU
# ----------------------------
maint_impact = 0.85 + 0.30 * np.sqrt(min(maint_base, 1000) / 1000)   # hypothèse modèle
life_new_actual = base_lifetime * maint_impact

bm = BRAND_MODIFIERS[brand]
tech_coeffs = {"Smartphone": 0.85, "Tablet": 0.85, "Switch / Router": 0.95, "Laptop": 0.80, "Room Screens": 0.80}
coeff_type = tech_coeffs.get(material_type, 0.80)

life_refurb_raw = life_new_actual * coeff_type * bm["lifetime"]
MAX_REFURB_OVER_NEW = 1.20
life_refurb_actual = min(life_refurb_raw, life_new_actual * MAX_REFURB_OVER_NEW)

# ----------------------------
# CALCULS FINANCIERS & CO2 (unitaires)
# ----------------------------
REFURB_PRICE_RATIO = 0.60
REFURB_PROD_RATIO = 0.25

tco_new = price + (energy_annual * life_new_actual * ELEC_PRICE) + maint_base
co2_new = carbon_prod + (energy_annual * life_new_actual * CO2_FACTOR)
annual_cost_new = tco_new / max(life_new_actual, 1e-9)

tco_refurb = (price * REFURB_PRICE_RATIO) + (energy_annual * life_refurb_actual * ELEC_PRICE) + (maint_base * bm["maint"])
co2_refurb = (carbon_prod * REFURB_PROD_RATIO * bm["carbon"]) + (energy_annual * life_refurb_actual * CO2_FACTOR)
annual_cost_refurb = tco_refurb / max(life_refurb_actual, 1e-9)

# Fleet totals (pour le type sélectionné)
fleet_annual_cost_new = annual_cost_new * fleet_size_current_type
fleet_annual_cost_refurb = annual_cost_refurb * fleet_size_current_type
fleet_savings_year = fleet_annual_cost_new - fleet_annual_cost_refurb

fleet_co2_new = co2_new * fleet_size_current_type
fleet_co2_refurb = co2_refurb * fleet_size_current_type
fleet_co2_savings = fleet_co2_new - fleet_co2_refurb

# ----------------------------
# SCORES (2 SCORES comparables) + cap 0..100
# ----------------------------
# Refurb vs New
score_fin_refurb = round(score_from_ratio(tco_refurb, tco_new), 1)
score_co2_refurb = round(score_from_ratio(co2_refurb, co2_new), 1)
roi_refurb = round(clip(0.6 * score_fin_refurb + 0.4 * score_co2_refurb, 0, 100), 1)

# New vs Refurb
score_fin_new = round(score_from_ratio(tco_new, tco_refurb), 1)
score_co2_new = round(score_from_ratio(co2_new, co2_refurb), 1)
roi_new = round(clip(0.6 * score_fin_new + 0.4 * score_co2_new, 0, 100), 1)

best = "Refurbished (Reconditionné)" if roi_refurb > roi_new else "New (Neuf)"
roi_best = max(roi_new, roi_refurb)
roi_color = "#10b981" if roi_best > 55 else "#f59e0b" if roi_best > 45 else "#ef4444"

# ----------------------------
# DISPLAY HERO
# ----------------------------
st.title("🌱 Green IT ROI Dashboard")
st.markdown(
    f"""<div class="roi-box" style="border-left-color: {roi_color};">
    <h1 style="color: {roi_color}; margin-bottom: 0;">ROI New: {roi_new} / 100 — ROI Refurb: {roi_refurb} / 100</h1>
    <p><b>Best option:</b> {best} — <b>Selected type:</b> {material_type} ({fleet_size_current_type} units)</p>
</div>""",
    unsafe_allow_html=True,
)

st.caption(f"✅ Scores capped 0–100 | roi_new={roi_new} | roi_refurb={roi_refurb}")

if life_refurb_raw > life_refurb_actual + 1e-9:
    st.warning("⚠️ Refurb lifetime a été plafonnée (garde-fou) pour éviter un résultat irréaliste.")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 🆕 New Equipment ({brand}) — {material_type}")
    st.metric("Annualized Cost (per device)", f"{annual_cost_new:.0f} €/year")
    st.caption(f"Estimated lifespan: {life_new_actual:.1f} years")
with col2:
    st.markdown(f"### ♻️ Refurbished ({brand}) — {material_type}")
    delta = ((annual_cost_refurb / max(annual_cost_new, 1e-9)) - 1) * 100
    st.metric("Annualized Cost (per device)", f"{annual_cost_refurb:.0f} €/year", delta=f"{delta:.1f}%")
    st.caption(f"Estimated lifespan: {life_refurb_actual:.1f} years")

# Fleet impact (pour le type sélectionné)
st.subheader("🏢 Fleet impact (for selected device type)")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.metric("Fleet annual cost (New)", f"{fleet_annual_cost_new:,.0f} €/year".replace(",", " "))
    st.metric("Fleet annual cost (Refurb)", f"{fleet_annual_cost_refurb:,.0f} €/year".replace(",", " "))
with fc2:
    st.metric("Fleet savings", f"{fleet_savings_year:,.0f} €/year".replace(",", " "))
with fc3:
    st.metric("Fleet CO₂ savings", f"{fleet_co2_savings:,.0f} kgCO₂e".replace(",", " "))

st.divider()

# ----------------------------
# GAUGES
# ----------------------------
def plot_custom_gauge(title, value):
    colors = ["#ef4444", "#f97316", "#fbbf24", "#10b981", "#059669"]
    labels = ["CRITIC", "LOW", "NEUTRAL", "PERFORMANT", "EXCELLENT"]

    fig = plt.figure(figsize=(10, 5), facecolor="none")
    ax = fig.add_subplot(projection="polar")
    num_segments = len(colors)
    segment_width = np.pi / num_segments

    for i, color in enumerate(colors):
        start_angle = np.pi - (i * segment_width)
        ax.bar(x=start_angle, width=-segment_width, height=1.0, bottom=2.0,
               color=color, edgecolor="white", linewidth=2, align="edge")
        text_angle = start_angle - (segment_width / 2)
        ax.annotate(labels[i], xy=(text_angle, 2.5), rotation=np.degrees(text_angle)-90,
                    color="#1e293b", fontweight="bold", ha="center", va="center", fontsize=8)

    needle_pos = np.pi - (min(value, 100) / 100 * np.pi)
    ax.annotate("", xy=(needle_pos, 2.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="wedge, tail_width=0.8", color="#1e293b", shrinkA=0))
    ax.annotate(f"{int(round(value))}", xy=(0, 0),
                bbox=dict(boxstyle="circle", facecolor="#1e293b", edgecolor="white", linewidth=3),
                fontsize=25, color="white", ha="center", va="center", fontweight="bold")

    plt.title(title, loc="center", pad=40, fontsize=14, fontweight="bold")
    ax.set_axis_off()
    ax.set_theta_zero_location("E")
    st.pyplot(fig)

g1, g2, g3, g4 = st.columns(4)
with g1: plot_custom_gauge("FIN New", score_fin_new)
with g2: plot_custom_gauge("FIN Refurb", score_fin_refurb)
with g3: plot_custom_gauge("CO₂ New", score_co2_new)
with g4: plot_custom_gauge("CO₂ Refurb", score_co2_refurb)

st.divider()
r1, r2 = st.columns(2)
with r1: plot_custom_gauge("ROI SCORE — NEW", roi_new)
with r2: plot_custom_gauge("ROI SCORE — REFURB", roi_refurb)

st.markdown(
    f"""
<div class="explanation-note">
    <b>Note stratégique :</b> Les scores sont calculés en <b>gain relatif</b> (50 = neutre, 0..100 borné). 
    Le score global pondère l'efficacité financière (60%) et l'impact carbone (40%).<br>
    <b>France / LVMH:</b> CO₂ réseau = {CO2_FACTOR:.4f} kgCO₂e/kWh ; Prix élec = {ELEC_PRICE:.2f} €/kWh.<br>
    Maintenance {maint_base}€ → durée de vie neuf {life_new_actual:.1f} ans ; refurb {life_refurb_actual:.1f} ans.
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# BAR CHART (legend fixed)
# ----------------------------
st.subheader("📊 Annual Cost Comparison (€ / year) — per device")
fig_bar, ax_bar = plt.subplots(figsize=(10, 2))
bars = ax_bar.barh(
    ["New Equipment", "Refurbished"],
    [annual_cost_new, annual_cost_refurb],
    color=["#94a3b8", roi_color],
    height=0.6,
)
for bar in bars:
    width = bar.get_width()
    ax_bar.text(width + 5, bar.get_y() + bar.get_height()/2, f"{width:.0f} €/year", va="center", fontweight="bold")

ax_bar.legend(
    handles=[
        Patch(facecolor="#94a3b8", label="New (Neuf)"),
        Patch(facecolor=roi_color, label="Refurbished (Reconditionné)"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=2,
    frameon=False,
)
ax_bar.set_axis_off()
fig_bar.tight_layout()
st.pyplot(fig_bar)

st.divider()

# ----------------------------
# DOCUMENTATION SECTION (parameters + sources/links)
# ----------------------------
st.header("📚 Documentation — parameters, values, justification & sources")

# Table "audit-ready"
doc_rows = [
    # User inputs
    {"Parameter": "price", "Value": price, "Unit": "€", "Category": "USER_INPUT",
     "Justification": "Purchase price per unit (new).", "Source": "User input (PO / invoice / procurement data)", "Link": ""},
    {"Parameter": "carbon_prod", "Value": carbon_prod, "Unit": "kgCO₂e", "Category": "USER_INPUT",
     "Justification": "Manufacturing + transport footprint proxy per unit.", "Source": "User input (PER/EPD/LCA supplier or internal baseline)", "Link": ""},
    {"Parameter": "base_lifetime", "Value": base_lifetime, "Unit": "years", "Category": "USER_INPUT",
     "Justification": "Theoretical service life (policy / warranty / manufacturer).", "Source": "User input (IT policy / warranty / DSI)", "Link": ""},
    {"Parameter": "energy_annual", "Value": energy_annual, "Unit": "kWh/year", "Category": "USER_INPUT",
     "Justification": "Annual electricity consumption per unit (usage dependent).", "Source": "User input (measurements / telemetry / assumption)", "Link": ""},
    {"Parameter": "maint_base", "Value": maint_base, "Unit": "€", "Category": "USER_INPUT",
     "Justification": "Maintenance budget for new devices (repair/support).", "Source": "User input (support contracts / SAV history)", "Link": ""},
    {"Parameter": "fleet_count_selected_type", "Value": fleet_size_current_type, "Unit": "devices", "Category": "USER_INPUT",
     "Justification": "Inventory count for selected device type.", "Source": "User input (asset inventory)", "Link": ""},
    {"Parameter": "fleet_count_total", "Value": fleet_size_total, "Unit": "devices", "Category": "USER_INPUT",
     "Justification": "Total inventory count across types.", "Source": "User input (asset inventory)", "Link": ""},

    # France / LVMH assumptions
    {"Parameter": "CO2_FACTOR", "Value": CO2_FACTOR, "Unit": "kgCO₂e/kWh", "Category": "SOURCED_DEFAULT",
     "Justification": "French grid carbon intensity (default).", "Source": "RTE — Bilan électrique 2024 (21.7 gCO₂e/kWh)", "Link": "https://www.rte-france.com/analyses-tendances-et-prospectives/bilan-electrique"},
    {"Parameter": "ELEC_PRICE", "Value": ELEC_PRICE, "Unit": "€/kWh", "Category": "ASSUMPTION_DEFAULT",
     "Justification": "Electricity price proxy (varies by contract).", "Source": "Default scenario; calibrate with LVMH contract prices", "Link": "https://www.cre.fr/Actualites"},
    
    # Refurb assumptions
    {"Parameter": "REFURB_PRICE_RATIO", "Value": REFURB_PRICE_RATIO, "Unit": "ratio", "Category": "ASSUMPTION",
     "Justification": "Refurb purchase price is assumed at 60% of new.", "Source": "Model assumption (replace with supplier/market data)", "Link": ""},
    {"Parameter": "REFURB_PROD_RATIO", "Value": REFURB_PROD_RATIO, "Unit": "ratio", "Category": "ASSUMPTION",
     "Justification": "Refurb manufacturing footprint assumed at 25% of new.", "Source": "Model assumption (replace with refurb LCA / supplier data)", "Link": ""},
    {"Parameter": "MAX_REFURB_OVER_NEW", "Value": MAX_REFURB_OVER_NEW, "Unit": "ratio", "Category": "MODEL_GUARDRAIL",
     "Justification": "Caps refurb lifetime to 120% of new to avoid unrealistic results.", "Source": "Guardrail (modeling choice)", "Link": ""},

    # Maintenance to lifetime function constants
    {"Parameter": "maint_impact_base", "Value": 0.85, "Unit": "ratio", "Category": "MODEL_ASSUMPTION",
     "Justification": "Baseline lifetime multiplier with minimal maintenance.", "Source": "Model assumption (calibrate with SAV)", "Link": ""},
    {"Parameter": "maint_impact_amplitude", "Value": 0.30, "Unit": "ratio", "Category": "MODEL_ASSUMPTION",
     "Justification": "Maximum additional lifetime multiplier from maintenance (sqrt).", "Source": "Model assumption (calibrate with SAV)", "Link": ""},
    {"Parameter": "maint_budget_cap", "Value": 1000, "Unit": "€", "Category": "MODEL_ASSUMPTION",
     "Justification": "Maintenance budget cap in lifetime function.", "Source": "Model assumption", "Link": ""},

    # Scoring weights
    {"Parameter": "weight_financial", "Value": 0.60, "Unit": "ratio", "Category": "GOVERNANCE",
     "Justification": "Weight of financial component in ROI score.", "Source": "Governance choice (project sponsor)", "Link": ""},
    {"Parameter": "weight_environmental", "Value": 0.40, "Unit": "ratio", "Category": "GOVERNANCE",
     "Justification": "Weight of CO₂ component in ROI score.", "Source": "Governance choice (project sponsor)", "Link": ""},
]

doc_df = pd.DataFrame(doc_rows)

# Show + export doc
st.dataframe(doc_df, use_container_width=True)

doc_csv = doc_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Export documentation (CSV)", doc_csv, "GreenIT_ROI_Documentation.csv", "text/csv")

# Links section (clickable)
st.subheader("🔗 Source links (to justify defaults & assumptions)")
st.markdown(
    """
- **RTE — Bilan électrique (France)** (grid carbon intensity, context): https://www.rte-france.com/analyses-tendances-et-prospectives/bilan-electrique  
- **CRE — Commission de Régulation de l'Énergie** (enterprise electricity price context): https://www.cre.fr/  
- **SDES — Prix de l’électricité** (statistics / price context): https://www.statistiques.developpement-durable.gouv.fr/  
- **Apple Product Environmental Reports** (example of device footprint sources): https://www.apple.com/environment/  
- **ADEME Base Empreinte** (emission factors / LCA resources): https://base-empreinte.ademe.fr/
"""
)

# ----------------------------
# EXPORT main report
# ----------------------------
v1, v2 = st.columns([2, 1])
with v1:
    if annual_cost_refurb < annual_cost_new:
        st.success(f"🏆 **PROVEN PROFITABILITY**: Refurbished saves you **{annual_cost_new - annual_cost_refurb:.0f} €/year** per unit.")
    else:
        st.warning("🚩 **PROFITABILITY ALERT**: High replacement frequency for refurbished offsets gain.")

with v2:
    df = pd.DataFrame(
        {
            "Indicator": [
                "Device Type",
                "Brand",
                "TCO",
                "Annual Cost",
                "CO2 Total",
                "Actual Life",
                "Fleet size (selected type)",
                "Fleet Annual Cost (selected type)",
                "Fleet CO2 Total (selected type)",
                "Elec price",
                "Grid CO2 factor",
                "Refurb price ratio",
                "Refurb prod CO2 ratio",
                "Max refurb vs new",
                "ROI score (NEW)",
                "ROI score (REFURB)",
            ],
            "New": [
                material_type,
                brand,
                tco_new,
                annual_cost_new,
                co2_new,
                life_new_actual,
                fleet_size_current_type,
                fleet_annual_cost_new,
                fleet_co2_new,
                ELEC_PRICE,
                CO2_FACTOR,
                1.0,
                1.0,
                1.0,
                roi_new,
                roi_refurb,
            ],
            "Refurbished": [
                material_type,
                brand,
                tco_refurb,
                annual_cost_refurb,
                co2_refurb,
                life_refurb_actual,
                fleet_size_current_type,
                fleet_annual_cost_refurb,
                fleet_co2_refurb,
                ELEC_PRICE,
                CO2_FACTOR,
                REFURB_PRICE_RATIO,
                REFURB_PROD_RATIO,
                MAX_REFURB_OVER_NEW,
                roi_new,
                roi_refurb,
            ],
        }
    )
    st.download_button("📥 Export report (CSV)", df.to_csv(index=False).encode("utf-8"), "GreenIT_ROI_Report.csv", "text/csv")
