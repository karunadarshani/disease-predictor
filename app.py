"""
app.py — Geospatial Disease Spread Predictor
Main Streamlit dashboard with interactive Folium maps.

Run with: streamlit run app.py
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.integrate import odeint
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import our project modules
from download_data   import setup_directories, download_district_geojson
from spatial_features import build_all_features
from disease_model   import sir_ode, simulate_all_districts, train_ml_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Disease Spread Predictor — India",
    page_icon  = "🦠",
    layout     = "wide",
)

# ── Cached data loading (only runs once per session) ─────────────────────────
@st.cache_data(show_spinner="Loading spatial data and running models…")
def load_data():
    setup_directories()
    gdf = download_district_geojson()
    gdf, X, feature_cols, adj = build_all_features(gdf)
    gdf = simulate_all_districts(gdf)
    gdf, ml_results = train_ml_model(gdf, feature_cols)
    return gdf, feature_cols, ml_results

gdf, feature_cols, ml = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦠 Disease Predictor")
    st.caption("India District-Level Analysis")
    st.divider()

    st.subheader("SIR Model Parameters")
    beta  = st.slider("Transmission rate β", 0.05, 0.60, 0.25, 0.01,
                      help="Higher β = more contagious disease")
    gamma = st.slider("Recovery rate γ",     0.03, 0.25, 0.08, 0.01,
                      help="Higher γ = faster recovery (~1/infectious days)")
    R0_val = beta / gamma
    st.metric("R₀ (reproduction number)", f"{R0_val:.2f}",
              delta="epidemic grows" if R0_val > 1 else "dies out",
              delta_color="inverse")

    seed_cases = st.slider("Seed cases",       1, 200,  10)
    sim_days   = st.slider("Simulation days", 30, 200, 120)

    st.divider()
    state_col = next((c for c in ["state","State","ST_NM","NAME_1"]
                      if c in gdf.columns), None)
    if state_col:
        all_states = sorted(gdf[state_col].dropna().unique().tolist())
        selected_states = st.multiselect("Filter by state", all_states,
                                          default=all_states[:5])
        gdf_view = gdf[gdf[state_col].isin(selected_states)] if selected_states else gdf
    else:
        gdf_view = gdf

    st.divider()
    st.caption("Built with GeoPandas · SciPy · Scikit-learn · Folium · Streamlit")

# ── Page title + KPI row ──────────────────────────────────────────────────────
st.title("🗺️ Geospatial Disease Spread Predictor — India")
st.markdown(
    "Predicts district-level epidemic risk using spatial features "
    "(population density, road proximity, adjacency, climate) + SIR epidemiological model."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Districts",         f"{len(gdf_view):,}")
c2.metric("High / Critical",   int((gdf_view["risk_score"] > 50).sum()))
c3.metric("Avg risk score",    f"{gdf_view['risk_score'].mean():.1f} / 100")
c4.metric("Model R²",          f"{ml['test_r2']:.3f}")
c5.metric("Features",          len(feature_cols))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_sir, tab_ml, tab_data = st.tabs([
    "🗺️ Risk Map", "🔬 SIR Simulator", "🤖 ML Model", "📋 Data"
])

# ════════════════════════════════════════════════════════════
# TAB 1: INTERACTIVE FOLIUM RISK MAP
# ════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Interactive District Risk Map")
    col_left, col_right = st.columns([4, 1])

    with col_right:
        map_var = st.radio("Colour map by", [
            "risk_score", "attack_rate", "ml_risk_pred",
            "pop_density", "road_proximity_km", "monsoon_index"
        ])
        color_scales = {
            "risk_score":        "RdYlGn_r",
            "attack_rate":       "OrRd",
            "ml_risk_pred":      "RdYlGn_r",
            "pop_density":       "Blues",
            "road_proximity_km": "YlOrBr",
            "monsoon_index":     "BuGn",
        }
        show_hover = st.checkbox("Show tooltips on hover", True)

    with col_left:
        # ── Build Folium choropleth ──
        m = folium.Map(
            location=[20.6, 78.9],
            zoom_start=5,
            tiles="CartoDB positron",
        )

        # Find the district name column (varies by GeoJSON source)
        name_col = next(
            (c for c in ["district", "DISTRICT", "dtname", "district_full", "NAME_2"]
             if c in gdf_view.columns),
            None
        )
        if name_col is None:
            gdf_view = gdf_view.copy()
            gdf_view["_id"] = gdf_view.index.astype(str)
            name_col = "_id"

        gdf_clean = gdf_view[gdf_view.geometry.notnull()].copy()

        folium.Choropleth(
            geo_data  = gdf_clean.__geo_interface__,
            data      = gdf_clean.reset_index(),
            columns   = [name_col, map_var],
            key_on    = f"feature.properties.{name_col}",
            fill_color    = color_scales.get(map_var, "RdYlGn_r"),
            fill_opacity  = 0.72,
            line_opacity  = 0.25,
            line_weight   = 0.4,
            legend_name   = map_var.replace("_", " ").title(),
            nan_fill_color= "lightgray",
        ).add_to(m)

        if show_hover:
            hover_cols = [c for c in [
                name_col, "risk_score", "attack_rate", "pop_density",
                "road_proximity_km", "n_neighbors", "risk_category"
            ] if c in gdf_clean.columns]

            folium.GeoJson(
                gdf_clean[hover_cols + ["geometry"]].__geo_interface__,
                style_function=lambda x: {"fillOpacity": 0, "weight": 0},
                tooltip=folium.GeoJsonTooltip(
                    fields  = hover_cols,
                    aliases = [c.replace("_", " ").title() for c in hover_cols],
                    sticky  = True,
                    localize= True,
                ),
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=740, height=520, returned_objects=[])

    # Risk distribution charts below map
    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        fig = px.histogram(
            gdf_view, x="risk_score", nbins=25,
            color_discrete_sequence=["#E74C3C"],
            title="Risk score distribution",
            labels={"risk_score": "Risk score (0–100)"},
        )
        fig.update_layout(height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        cat_counts = gdf_view["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            cat_counts, values="Count", names="Category",
            color="Category",
            color_discrete_map={
                "Low":"#27AE60", "Moderate":"#F39C12",
                "High":"#E67E22", "Critical":"#E74C3C"
            },
            title="Districts by risk category",
        )
        fig2.update_layout(height=260)
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2: INTERACTIVE SIR SIMULATOR
# ════════════════════════════════════════════════════════════
with tab_sir:
    st.subheader("SIR Epidemic Model — Interactive Simulator")

    st.info("""
    **How the SIR model works:** The population is split into 3 groups.
    S (Susceptible) → get infected at rate β·S·I/N → become I (Infected) →
    recover at rate γ → become R (Recovered/immune).
    **R₀ = β/γ** is the average number of people each infected person infects.
    R₀ > 1 → epidemic grows. R₀ < 1 → fizzles out.
    """)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        pop = st.number_input("Population", 10_000, 10_000_000, 1_000_000, 50_000)
    with sc2:
        pd_val = st.slider("Pop density (per km²)", 50, 8000, 800)
    with sc3:
        rc_val = st.slider("Road connectivity", 0.0, 1.0, 0.6, 0.05)

    # Compute spatial beta modifier
    beta_spatial = beta * (1 + 0.3 * min(pd_val/1000, 2) + 0.2 * rc_val)
    st.caption(f"Effective β (with spatial factors) = {beta_spatial:.3f} | "
               f"R₀ = {beta_spatial/gamma:.2f}")

    t   = np.linspace(0, sim_days, sim_days * 3)
    I0  = seed_cases
    sol = odeint(sir_ode, [pop - I0, I0, 0], t, args=(beta_spatial, gamma))
    S_arr, I_arr, R_arr = sol.T

    fig_sir = go.Figure()
    fig_sir.add_trace(go.Scatter(x=t, y=S_arr/pop*100, name="Susceptible (S)",
                                  line=dict(color="#3498DB", width=2.5)))
    fig_sir.add_trace(go.Scatter(x=t, y=I_arr/pop*100, name="Infected (I)",
                                  line=dict(color="#E74C3C", width=2.5),
                                  fill="tozeroy",
                                  fillcolor="rgba(231,76,60,0.08)"))
    fig_sir.add_trace(go.Scatter(x=t, y=R_arr/pop*100, name="Recovered (R)",
                                  line=dict(color="#27AE60", width=2.5)))

    pk = int(np.argmax(I_arr) / 3)
    fig_sir.add_vline(x=pk, line_dash="dash", line_color="gray",
                      annotation_text=f"Peak: day {pk}", annotation_position="top right")

    fig_sir.update_layout(
        title=f"SIR Simulation — R₀={beta_spatial/gamma:.2f}  |  Pop={pop:,}",
        xaxis_title="Days",
        yaxis_title="% of population",
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_sir, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R₀ effective",  f"{beta_spatial/gamma:.2f}")
    m2.metric("Peak infected", f"{I_arr.max()/pop*100:.1f}%")
    m3.metric("Days to peak",  f"{pk}")
    m4.metric("Attack rate",   f"{R_arr[-1]/pop*100:.1f}%")

# ════════════════════════════════════════════════════════════
# TAB 3: ML MODEL RESULTS
# ════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("Random Forest Model Performance")

    ml1, ml2 = st.columns(2)

    with ml1:
        st.markdown("**Performance metrics**")
        perf = pd.DataFrame({
            "Metric": ["Train R²", "Test R²", "CV R² (5-fold)", "Test MAE"],
            "Value": [
                f"{ml['train_r2']:.4f}",
                f"{ml['test_r2']:.4f}",
                f"{ml['cv_mean']:.4f} ± {ml['cv_std']:.4f}",
                f"{ml['test_mae']:.2f} pts",
            ]
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("**What R² means:**")
        st.markdown("- R² = 1.0 → perfect predictions")
        st.markdown("- R² = 0.85 → model explains 85% of variance in risk scores")
        st.markdown("- R² = 0.0 → no better than predicting the mean")

    with ml2:
        # Predicted vs actual
        fig_pva = px.scatter(
            x=ml["y_test"], y=ml["y_pred"],
            labels={"x": "Actual risk score", "y": "Predicted risk score"},
            title="Predicted vs actual (test set)",
            opacity=0.55,
            color_discrete_sequence=["#E74C3C"],
        )
        mx = max(float(ml["y_test"].max()), float(ml["y_pred"].max()))
        fig_pva.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                          line=dict(color="black", dash="dash", width=1))
        fig_pva.update_layout(height=320)
        st.plotly_chart(fig_pva, use_container_width=True)

    st.divider()
    st.subheader("Feature Importance")

    fi = ml["feature_importance"]
    fig_fi = px.bar(
        fi, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="RdYlGn",
        title="Which features drive disease spread risk most?",
    )
    fig_fi.update_layout(
        height=360,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Scatter: top feature vs risk
    top_feat = fi.iloc[0]["feature"]
    fig_sc = px.scatter(
        gdf, x=top_feat, y="risk_score",
        color="risk_category",
        color_discrete_map={
            "Low":"#27AE60","Moderate":"#F39C12",
            "High":"#E67E22","Critical":"#E74C3C"
        },
        opacity=0.55,
        trendline="ols",
        title=f"Top feature: {top_feat} vs risk score",
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4: DATA TABLE
# ════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Full District Dataset")

    display_cols = [c for c in [
        "district", "state", "population", "area_km2", "pop_density",
        "road_proximity_km", "road_connectivity", "n_neighbors",
        "monsoon_index", "attack_rate", "peak_day",
        "risk_score", "risk_category", "ml_risk_pred",
    ] if c in gdf.columns]

    cat_filter = st.multiselect(
        "Filter by risk category", ["Low","Moderate","High","Critical"],
        default=["High","Critical"]
    )
    df_show = gdf[display_cols]
    if cat_filter and "risk_category" in df_show.columns:
        df_show = df_show[df_show["risk_category"].isin(cat_filter)]

    st.dataframe(
        df_show.sort_values("risk_score", ascending=False)
               .head(300).round(3),
        use_container_width=True,
    )

    csv = df_show.to_csv(index=False)
    st.download_button("📥 Download as CSV", csv,
                       "disease_risk_results.csv", "text/csv")