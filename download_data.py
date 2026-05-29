"""
download_data.py
Downloads real India district boundaries and population data.
Data source: geohacker/india (GitHub) — open data, no API key needed.
"""

import os
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# ── Real GeoJSON source URLs (tries each in order) ──────────────────────────
DISTRICT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
]

def setup_directories():
    """Create project folder structure."""
    for folder in ["data/raw", "data/processed", "outputs"]:
        os.makedirs(folder, exist_ok=True)
    print("✓ Folder structure created")


def download_district_geojson():
    """
    Download India district boundaries from open GitHub sources.
    If download fails, falls back to synthetic India-like districts.
    
    Returns: GeoDataFrame with one row per district
    """
    save_path = "data/raw/india_districts.geojson"

    # Already downloaded? Load from disk.
    if os.path.exists(save_path):
        print("✓ GeoJSON already on disk — loading...")
        gdf = gpd.read_file(save_path)
        print(f"  {len(gdf)} districts loaded")
        return gdf

    # Try each URL
    for url in DISTRICT_GEOJSON_URLS:
        try:
            print(f"Downloading India district boundaries...\n  {url}")
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                gdf = gpd.read_file(save_path)
                if len(gdf) > 10:
                    print(f"✓ Downloaded successfully! {len(gdf)} districts")
                    return gdf
        except Exception as e:
            print(f"  ✗ Failed ({e})")

    # Fallback: generate synthetic India-like districts
    print("⚠ Download failed. Creating synthetic dataset with real state structures...")
    return _create_synthetic_districts(save_path)


def _create_synthetic_districts(save_path):
    """
    Creates realistic synthetic India districts if internet download fails.
    Uses real Indian state names, real approximate coordinates (Census 2011),
    and realistic population figures. This is only a fallback.
    """
    np.random.seed(42)

    # Real Indian state data: (state, center_lat, center_lon, num_districts, pop_million)
    STATES = [
        ("Uttar Pradesh",      26.85, 80.95, 18, 199),
        ("Maharashtra",        19.75, 75.71, 16, 112),
        ("Bihar",              25.10, 85.31, 12, 104),
        ("West Bengal",        22.99, 87.85, 10,  91),
        ("Madhya Pradesh",     22.97, 78.66, 14,  72),
        ("Tamil Nadu",         11.13, 78.66, 10,  72),
        ("Rajasthan",          27.02, 74.22, 12,  68),
        ("Karnataka",          15.32, 75.71, 10,  61),
        ("Gujarat",            22.26, 71.19,  8,  60),
        ("Andhra Pradesh",     15.91, 79.74,  8,  49),
        ("Odisha",             20.94, 84.80,  7,  42),
        ("Telangana",          18.11, 79.02,  6,  35),
        ("Jharkhand",          23.61, 85.28,  6,  33),  # Karuna's home state!
        ("Kerala",             10.85, 76.27,  5,  33),
        ("Assam",              26.20, 92.94,  5,  31),
        ("Punjab",             31.15, 75.34,  5,  28),
        ("Chhattisgarh",       21.28, 81.87,  6,  26),
        ("Haryana",            29.06, 76.09,  5,  25),
        ("Uttarakhand",        30.07, 79.07,  4,  10),
        ("Himachal Pradesh",   31.10, 77.17,  3,   7),
    ]

    districts = []
    dist_id = 1

    for state_name, c_lat, c_lon, n_dist, state_pop_m in STATES:
        for i in range(n_dist):
            # Scatter districts around state center
            d_lat = c_lat + np.random.uniform(-2.2, 2.2)
            d_lon = c_lon + np.random.uniform(-2.2, 2.2)
            half  = np.random.uniform(0.28, 0.48)  # polygon half-size in degrees

            polygon   = box(d_lon - half, d_lat - half, d_lon + half, d_lat + half)

            # Population: state total / number of districts ± noise
            pop = int((state_pop_m * 1e6 / n_dist) * np.random.uniform(0.4, 1.8))

            districts.append({
                "district_id":   dist_id,
                "district":      f"{state_name.split()[0]}_D{i+1:02d}",
                "district_full": f"{state_name} District {i+1}",
                "state":         state_name,
                "population":    pop,
                "geometry":      polygon,
            })
            dist_id += 1

    gdf = gpd.GeoDataFrame(districts, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(save_path, driver="GeoJSON")
    print(f"✓ Created {len(gdf)} synthetic districts")
    return gdf


def get_real_jharkhand_population():
    """
    Real Census 2011 population for all 24 Jharkhand districts.
    Source: Census of India 2011 (public data).
    Useful if you want to filter to your home state specifically.
    """
    return {
        "Ranchi":              2914253,
        "Dhanbad":             2682662,
        "Bokaro":              2062330,
        "Giridih":             2445203,
        "East Singhbhum":      2291032,
        "Hazaribagh":          1734005,
        "Deoghar":             1492073,
        "Garhwa":              1322387,
        "Palamu":              1936319,
        "Godda":               1313551,
        "Chatra":              1042886,
        "Koderma":              717169,
        "Latehar":              731249,
        "Lohardaga":            461790,
        "Gumla":               1025213,
        "Simdega":              599813,
        "West Singhbhum":      1502338,
        "Saraikela Kharsawan": 1065056,
        "Jamtara":              791042,
        "Dumka":               1321442,
        "Pakur":                899200,
        "Sahebganj":           1150038,
        "Ramgarh":              949159,
        "Khunti":               531885,
    }


if __name__ == "__main__":
    setup_directories()
    gdf = download_district_geojson()
    print("\nSample columns:", list(gdf.columns))
    print("CRS:", gdf.crs)
    print(gdf.head(3).to_string())