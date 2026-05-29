"""
disease_model.py

Part A: SIR epidemiological model
  → Simulates disease spread in each district using differential equations
  → Generates the 'risk score' target variable for ML

Part B: Random Forest ML model
  → Learns to predict risk score from spatial features
  → Outputs feature importance (which factors drive spread?)
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint          # solves differential equations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
# PART A: SIR EPIDEMIOLOGICAL MODEL
# ════════════════════════════════════════════════════════════

def sir_ode(compartments, t, beta, gamma):
    """
    The SIR model as differential equations.

    Compartments:
      S = Susceptible  (healthy, can catch disease)
      I = Infected     (currently sick and infectious)
      R = Recovered    (immune, or deceased)

    Parameters:
      beta  = transmission rate (how fast S → I)
              Depends on: contact rate × probability of transmission
      gamma = recovery rate (how fast I → R)
              gamma = 1 / infectious_period (COVID: ~10 days → gamma ≈ 0.1)

    Key number: R0 = beta / gamma (Basic Reproduction Number)
      R0 > 1 → epidemic grows (each person infects >1 other)
      R0 < 1 → epidemic dies out
      COVID original strain: R0 ≈ 2.5 (beta=0.25, gamma=0.1)
      Delta variant:         R0 ≈ 5–6
      Measles:               R0 ≈ 12–18
    """
    S, I, R = compartments
    N = S + I + R       # total population (conserved)

    dS_dt = -beta * S * I / N           # S decreases as infections happen
    dI_dt =  beta * S * I / N - gamma * I   # I grows from new infections, shrinks from recovery
    dR_dt =  gamma * I                  # R grows as infected people recover

    return [dS_dt, dI_dt, dR_dt]


def simulate_one_district(population, pop_density, road_connectivity,
                           monsoon_index, seed_cases=5, days=120):
    """
    Run SIR simulation for a single district.

    KEY IDEA: beta is not fixed — it varies by district spatial features.
    High-density urban districts have more daily contacts → higher beta.
    Well-connected highway districts have more inter-district movement → higher beta.
    Monsoon causes indoor crowding → slightly higher beta.

    This links geography → epidemiology — that's what makes this project special!
    """
    # Base transmission rate (COVID-19 delta-like)
    beta_base = 0.22
    gamma     = 0.08   # ~12-day infectious period

    # Spatial modifiers (each adds ≤30% to beta)
    density_factor       = 0.30 * min(pop_density / 1000, 2.0)     # more crowded → higher beta
    connectivity_factor  = 0.20 * road_connectivity                  # more roads → more mixing
    climate_factor       = 0.10 * monsoon_index                      # humid → more indoor contact

    beta = beta_base * (1 + density_factor + connectivity_factor + climate_factor)

    # Initial conditions
    I0 = min(seed_cases, int(population * 0.0005))
    S0 = population - I0
    R0_init = 0

    # Time array (in days)
    t = np.linspace(0, days, days * 2)

    # Solve the ODE system using scipy
    # odeint integrates the differential equations numerically (Runge-Kutta)
    solution = odeint(sir_ode, [S0, I0, R0_init], t, args=(beta, gamma))
    S_arr, I_arr, R_arr = solution.T

    peak_infected = float(I_arr.max())
    peak_day      = int(I_arr.argmax() / 2)   # divide by 2 because of doubled time points
    total_cases   = float(R_arr[-1] + I_arr[-1])
    attack_rate   = total_cases / population    # fraction of population eventually infected
    R0_value      = beta / gamma

    return {
        "peak_infected": peak_infected,
        "peak_day":      peak_day,
        "attack_rate":   attack_rate,
        "total_cases":   total_cases,
        "R0_effective":  round(R0_value, 3),
    }


def simulate_all_districts(gdf):
    """
    Run the SIR model for every district and compute a composite risk score.

    risk_score (0–100) combines:
    - attack_rate (60% weight): what fraction of population gets infected
    - speed       (40% weight): how fast the epidemic peaks (faster = higher risk)
    """
    print("\nRunning SIR simulation for all districts...")
    results = []

    for idx, row in gdf.iterrows():
        sim = simulate_one_district(
            population        = row.get("population", 500_000),
            pop_density       = row.get("pop_density", 300),
            road_connectivity = row.get("road_connectivity", 0.5),
            monsoon_index     = row.get("monsoon_index", 0.5),
        )
        results.append(sim)

    sim_df = pd.DataFrame(results, index=gdf.index)
    for col in sim_df.columns:
        gdf[col] = sim_df[col]

    # Composite risk score (0–100)
    ar_norm   = (gdf["attack_rate"] - gdf["attack_rate"].min()) / \
                (gdf["attack_rate"].max() - gdf["attack_rate"].min() + 1e-9)
    spd_norm  = 1 - (gdf["peak_day"] - gdf["peak_day"].min()) / \
                    (gdf["peak_day"].max()  - gdf["peak_day"].min()  + 1e-9)

    gdf["risk_score"] = (0.6 * ar_norm + 0.4 * spd_norm) * 100

    gdf["risk_category"] = pd.cut(
        gdf["risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Critical"],
        include_lowest=True
    )

    print(f"✓ Simulated {len(gdf)} districts")
    print("Risk distribution:")
    print(gdf["risk_category"].value_counts().sort_index().to_string())
    return gdf


# ════════════════════════════════════════════════════════════
# PART B: MACHINE LEARNING — RANDOM FOREST
# ════════════════════════════════════════════════════════════

def train_ml_model(gdf, feature_cols):
    """
    Train a Random Forest to predict risk_score from spatial features.

    Random Forest:
    - Builds 200 independent decision trees on random subsets of data
    - Each tree asks questions like:
        "pop_density > 500?" → "road_proximity < 80 km?" → risk = 72
    - Final prediction = average of all 200 trees
    - Feature importance = how often a feature reduces prediction error

    Why no StandardScaler?
    Random Forests are tree-based — they don't use distances or dot products.
    They split on feature values, so scale doesn't matter.
    (Unlike K-Means or SVMs, which need scaling!)
    """
    print("\n" + "="*55)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*55)

    X = gdf[feature_cols].copy().fillna(gdf[feature_cols].median())
    y = gdf["risk_score"].copy()

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators  = 200,    # 200 trees in the forest
        max_depth     = 12,     # max depth per tree (prevents overfitting)
        min_samples_leaf = 3,   # each leaf needs at least 3 districts
        random_state  = 42,
        n_jobs        = -1      # use all CPU cores (faster)
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_r2     = r2_score(y_train, y_pred_train)
    test_r2      = r2_score(y_test,  y_pred_test)
    test_mae     = mean_absolute_error(y_test, y_pred_test)

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print(f"\nPerformance:")
    print(f"  Train R²   = {train_r2:.4f}")
    print(f"  Test  R²   = {test_r2:.4f}  ← main metric")
    print(f"  CV R²      = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test MAE   = {test_mae:.2f} risk-score points")

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop Feature Importances:")
    for _, row in fi.head(5).iterrows():
        bar = "█" * int(row["importance"] * 60)
        print(f"  {row['feature']:22s}  {bar} {row['importance']:.3f}")

    # Add ML predictions to GeoDataFrame
    gdf["ml_risk_pred"] = model.predict(X)

    return gdf, {
        "model":       model,
        "feature_cols": feature_cols,
        "train_r2":    train_r2,
        "test_r2":     test_r2,
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "test_mae":    test_mae,
        "feature_importance": fi,
        "X_test":      X_test,
        "y_test":      y_test,
        "y_pred":      y_pred_test,
    }


if __name__ == "__main__":
    from download_data import setup_directories, download_district_geojson
    from spatial_features import build_all_features

    setup_directories()
    gdf = download_district_geojson()
    gdf, X, feats, adj = build_all_features(gdf)
    gdf = simulate_all_districts(gdf)
    gdf, results = train_ml_model(gdf, feats)

    save_cols = [c for c in gdf.columns if c not in ["geometry"] or True]
    gdf.to_file("data/processed/final_results.geojson", driver="GeoJSON")
    print("\n✓ Saved final_results.geojson")