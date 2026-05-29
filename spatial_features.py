"""
spatial_features.py
Extracts 8 geographic features from district boundaries.
Libraries: GeoPandas (spatial operations), Shapely (geometry math)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import warnings
warnings.filterwarnings('ignore')


# ── 1. Area and Centroid ──────────────────────────────────────────────────────

def calculate_area_and_centroid(gdf):
    """
    Area in km² and centroid (lat, lon) for each district.

    Why project to EPSG:32644?
    The default CRS (EPSG:4326) stores coordinates in degrees.
    Degree-based area is meaningless (a degree at the equator ≠ at the poles).
    EPSG:32644 is UTM Zone 44N — it covers India and stores coordinates in metres,
    so area comes out in m² which we convert to km².
    """
    # Project to metres for accurate area
    gdf_metres = gdf.to_crs("EPSG:32644")
    gdf["area_km2"] = (gdf_metres.geometry.area / 1_000_000).round(2)

    # Centroid in WGS84 (degrees) — useful for distance calculations
    centroids = gdf.geometry.centroid   # Shapely Point objects
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x

    print(f"✓ Area + centroid: {gdf['area_km2'].min():.0f}–{gdf['area_km2'].max():.0f} km²")
    return gdf


# ── 2. Population Density ─────────────────────────────────────────────────────

def add_population_density(gdf):
    """
    Population density = population / area_km²

    This is THE most important disease-spread predictor.
    Dense urban areas have more person-to-person contacts → faster spread.

    We also compute log_pop_density because density is heavily right-skewed
    (Mumbai ~20,000/km² vs Leh ~3/km²). Log scale makes it ML-friendly.
    """
    # Ensure we have a population column
    if "population" not in gdf.columns:
        print("  No population column found — generating approximate values...")
        np.random.seed(42)
        base = 300 + 200 * np.random.lognormal(0, 0.8, len(gdf))
        gdf["population"] = (gdf["area_km2"] * base).astype(int).clip(50_000, 5_000_000)

    gdf["pop_density"]      = (gdf["population"] / gdf["area_km2"]).round(2)
    gdf["log_pop_density"]  = np.log1p(gdf["pop_density"]).round(4)

    print(f"✓ Population density: {gdf['pop_density'].min():.0f}–{gdf['pop_density'].max():.0f} /km²")
    return gdf


# ── 3. Road Proximity ─────────────────────────────────────────────────────────

# Real coordinates of major National Highway junctions across India.
# These act as proxy nodes for highway accessibility.
# Source: National Highway Authority of India route map (public knowledge).
HIGHWAY_NODES_INDIA = [
    # NH-44 (Srinagar–Kanyakumari — longest highway in India)
    (28.704, 77.103),   # Delhi
    (26.847, 80.946),   # Lucknow
    (17.385, 78.487),   # Hyderabad
    (13.083, 80.271),   # Chennai
    (8.524,  76.937),   # Thiruvananthapuram

    # East–West corridor
    (19.076, 72.878),   # Mumbai
    (23.023, 72.571),   # Ahmedabad
    (21.146, 79.088),   # Nagpur (geometric centre of India)
    (18.520, 73.857),   # Pune
    (22.572, 88.364),   # Kolkata

    # North India
    (30.733, 76.779),   # Chandigarh
    (26.912, 75.787),   # Jaipur
    (25.594, 85.138),   # Patna
    (27.354, 88.612),   # Siliguri area

    # Eastern India
    (20.296, 85.825),   # Bhubaneswar
    (23.610, 85.280),   # Ranchi  ← Jharkhand capital (your area!)
    (22.987, 87.855),   # Kharagpur

    # South India
    (12.972, 77.595),   # Bengaluru
    (10.851, 76.271),   # Coimbatore
    (9.931,  76.267),   # Kochi
    (15.317, 75.714),   # Hubballi
    (16.506, 80.648),   # Vijayawada

    # Central hubs
    (24.585, 73.713),   # Udaipur
    (25.436, 81.846),   # Prayagraj
    (26.450, 80.332),   # Kanpur
    (22.720, 75.858),   # Indore
    (23.260, 77.413),   # Bhopal
    (21.251, 81.630),   # Raipur
    (26.761, 83.373),   # Gorakhpur
    (27.177, 78.008),   # Agra
]

def calculate_road_proximity(gdf):
    """
    Distance (km) from each district centroid to the nearest National Highway node.

    How it works:
    1. We create a Shapely MultiPoint from all highway junction coordinates.
    2. For each district centroid, we use Shapely's nearest_points() to find
       the closest highway junction.
    3. The Euclidean distance in degrees is multiplied by 111 (km per degree).

    Result:
    - Short distance → district is near a highway → people travel more → faster spread
    - Long distance → remote/rural → slower spread via movement
    - road_connectivity (0–1): inverse score — higher = better connected
    """
    highway_multipoint = MultiPoint([Point(lon, lat) for lat, lon in HIGHWAY_NODES_INDIA])

    def dist_to_nearest_highway(centroid_point):
        # nearest_points returns (point_on_geom1, point_on_geom2)
        nearest = nearest_points(centroid_point, highway_multipoint)[1]
        dist_deg = centroid_point.distance(nearest)
        return dist_deg * 111  # approx km

    print("Calculating road proximity for all districts...")
    centroids = gdf.geometry.centroid
    gdf["road_proximity_km"] = centroids.apply(dist_to_nearest_highway).round(2)

    # Normalize to 0–1 connectivity score (1 = very close to highway)
    gdf["road_connectivity"] = (1 / (1 + gdf["road_proximity_km"] / 60)).round(4)

    print(f"✓ Road proximity: {gdf['road_proximity_km'].min():.1f}–{gdf['road_proximity_km'].max():.1f} km")
    return gdf


# ── 4. District Adjacency ─────────────────────────────────────────────────────

def calculate_adjacency(gdf):
    """
    Count how many districts share a border with each district.

    Why this matters epidemiologically:
    Adjacent districts can transmit disease across the border.
    A district with many neighbors is more exposed to disease arriving from outside.

    Method:
    We use GeoPandas spatial join with a tiny buffer (0.01°≈1 km).
    The buffer ensures that districts sharing only a thin border line
    (which might not "touch" due to floating point precision) still count as neighbors.
    """
    print("Computing district adjacency matrix (this takes ~30 seconds for 700 districts)...")

    gdf_buf = gdf.copy()
    gdf_buf["geometry"] = gdf_buf.geometry.buffer(0.01)  # ~1 km buffer

    # Spatial join: find which district polygons intersect each other
    joined = gpd.sjoin(
        gdf_buf[["geometry"]].reset_index(names="left_idx"),
        gdf_buf[["geometry"]].reset_index(names="right_idx"),
        how="inner",
        predicate="intersects"
    )
    joined = joined[joined["left_idx"] != joined["right_idx"]]  # drop self-matches

    neighbor_counts = joined.groupby("left_idx")["right_idx"].count()
    gdf["n_neighbors"] = gdf.index.map(neighbor_counts).fillna(0).astype(int)

    print(f"✓ Adjacency: avg {gdf['n_neighbors'].mean():.1f} neighbors per district")
    return gdf, joined


# ── 5. Climate Features ───────────────────────────────────────────────────────

def add_climate_features(gdf):
    """
    Climate-based disease transmission features derived purely from coordinates.
    No external API needed — latitude and longitude encode climate well.

    monsoon_index: India's monsoon affects indoor crowding and waterborne disease.
      Higher in eastern coastal regions (Kerala, West Bengal) — real geographic pattern.
    temp_index: Tropical southern India is warmer → different pathogen survival.
    climate_zone: Simple 4-zone classification based on latitude.
    """
    lat = gdf["centroid_lat"]
    lon = gdf["centroid_lon"]

    # Monsoon exposure: stronger in east and coastal south (real meteorological pattern)
    gdf["monsoon_index"] = (
        0.4 * (1 - (lat - 18).abs() / 18).clip(0, 1) +  # latitude: peaks at 18°N
        0.6 * ((lon - 68) / 30).clip(0, 1)               # longitude: increases eastward
    ).round(4)

    # Temperature proxy: southern tropical districts are warmer
    gdf["temp_index"] = (1 - (lat - 8) / 30).clip(0, 1).round(4)

    # 4-zone classification
    gdf["climate_zone"] = pd.cut(
        lat,
        bins=[0, 15, 23, 30, 40],
        labels=["tropical", "humid_subtropical", "subtropical", "temperate"]
    ).astype(str)

    # Encode climate zone as integer for ML
    zone_map = {"tropical": 3, "humid_subtropical": 2, "subtropical": 1, "temperate": 0}
    gdf["climate_zone_enc"] = gdf["climate_zone"].map(zone_map).fillna(1).astype(int)

    print("✓ Climate features: monsoon_index, temp_index, climate_zone")
    return gdf


# ── Master Function ───────────────────────────────────────────────────────────

def build_all_features(gdf):
    """
    Run all feature engineering steps and return clean feature matrix.
    """
    print("\n" + "="*55)
    print("SPATIAL FEATURE ENGINEERING")
    print("="*55)

    gdf = calculate_area_and_centroid(gdf)
    gdf = add_population_density(gdf)
    gdf = calculate_road_proximity(gdf)
    gdf, adj_df = calculate_adjacency(gdf)
    gdf = add_climate_features(gdf)

    FEATURE_COLS = [
        "pop_density",        # people / km²
        "log_pop_density",    # log-transformed (handles skew)
        "area_km2",           # district size
        "road_proximity_km",  # km to nearest highway
        "road_connectivity",  # 0–1 highway access score
        "n_neighbors",        # number of adjacent districts
        "monsoon_index",      # 0–1 climate moisture
        "temp_index",         # 0–1 temperature proxy
        "climate_zone_enc",   # 0–3 categorical
        "centroid_lat",       # geographic position
        "centroid_lon",
    ]

    available = [c for c in FEATURE_COLS if c in gdf.columns]
    X = gdf[available].fillna(gdf[available].median(numeric_only=True))

    print(f"\n✓ Feature matrix: {X.shape[0]} districts × {X.shape[1]} features")
    return gdf, X, available, adj_df


if __name__ == "__main__":
    from download_data import setup_directories, download_district_geojson
    setup_directories()
    gdf = download_district_geojson()
    gdf, X, feats, adj = build_all_features(gdf)
    gdf.to_file("data/processed/features.geojson", driver="GeoJSON")
    X.to_csv("data/processed/feature_matrix.csv", index=False)
    print("\nSaved processed data!")