import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
MASTER_CSV = "./data2226/groundtruth2226.csv"
VALID_YEARS = [2022, 2023, 2024] # Excluding 2026
OUTPUT_CSV = "./empties2226/empties2226.csv"

def load_zero_data(years):
    """Helper to load and concatenate zero CSVs across years."""
    data_frames = []
    for year in years:
        file_path = Path(f"{year}/{year}_zero.csv")
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['year'] = year
            data_frames.append(df)
        else:
            print(f"Warning: {file_path} not found.")
    
    if not data_frames:
        raise FileNotFoundError("No _zero.csv files found.")
        
    combined = pd.concat(data_frames, ignore_index=True)
    
    # Normalize column names
    combined.columns = [c.lower() for c in combined.columns]
    rename_map = {
        'ship_latitude': 'latitude',
        'ship_longitude': 'longitude',
        'image_name': 'imagename',
        'altimeter_altitude_meter': 'altitude'
    }
    combined.rename(columns=rename_map, inplace=True)
    return combined

# ==========================================
# 1. Calculate Targets & Edges from Positives
# ==========================================
print("Loading sanitized master positive annotations...")
pos_df = pd.read_csv(MASTER_CSV)

# Filter down to the target years for empty-sourcing
pos_target_years = pos_df[pos_df['year'].isin(VALID_YEARS)].copy()

# Get unique images to calculate proportions without density bias
pos_images = pos_target_years.drop_duplicates(subset=['imagename']).copy()
pos_images_all = pos_df.drop_duplicates(subset=['imagename']).copy()

# Calculate the 15% target dynamically
target_total_empties = int(len(pos_images_all) * 0.15)
print(f"Dynamic Target: {target_total_empties} empty images (15% of 2022-2026 positives).")

# Calculate exact bin edges from the clean positive footprints
lat_bins, lat_edges = pd.qcut(pos_images['latitude'], q=4, retbins=True, duplicates="drop")
lon_bins, lon_edges = pd.qcut(pos_images['longitude'], q=4, retbins=True, duplicates="drop")

# Apply bins and create stratum identifier
pos_images['lat_bin'] = pd.cut(pos_images['latitude'], bins=lat_edges, include_lowest=True)
pos_images['lon_bin'] = pd.cut(pos_images['longitude'], bins=lon_edges, include_lowest=True)
pos_images['stratum'] = pos_images['year'].astype(str) + "_" + pos_images['lat_bin'].astype(str) + "_" + pos_images['lon_bin'].astype(str)

# Calculate exact integer target for each stratum
stratum_counts = pos_images.groupby('stratum').size().reset_index(name='pos_count')
total_valid_positives = stratum_counts['pos_count'].sum()
stratum_counts['target_zeros'] = ((stratum_counts['pos_count'] / total_valid_positives) * target_total_empties).round().astype(int)

# ==========================================
# 2. Load, Sanitize, and Map the Zeros
# ==========================================
print("\nLoading and sanitizing zero images...")
zero_df = load_zero_data(VALID_YEARS)
initial_zero_count = len(zero_df)

# Sanitize zeros using the same logic as the positives
zero_df = zero_df.dropna(subset=['latitude', 'longitude']).drop_duplicates(subset=['imagename']).copy()
zero_df = zero_df[(zero_df['longitude'] >= -76) & (zero_df['longitude'] <= -66)]

if 'altitude' in zero_df.columns:
    zero_df = zero_df[zero_df['altitude'] <= 3]

print(f"Sanitized zeros from {initial_zero_count} to {len(zero_df)} valid candidate images.")

# Apply the exact same positive bin edges to the zeros
# Note: Zeros outside the scallop bounding box will receive NaN and be naturally filtered out
zero_df['lat_bin'] = pd.cut(zero_df['latitude'], bins=lat_edges, include_lowest=True)
zero_df['lon_bin'] = pd.cut(zero_df['longitude'], bins=lon_edges, include_lowest=True)
zero_df['stratum'] = zero_df['year'].astype(str) + "_" + zero_df['lat_bin'].astype(str) + "_" + zero_df['lon_bin'].astype(str)

# Drop zeros that fell outside the positive spatial bounds (NaN bins)
zero_df = zero_df.dropna(subset=['lat_bin', 'lon_bin'])

# ==========================================
# 3. Sample Zeros by Stratum
# ==========================================
print("\nSampling zeros to match spatial/temporal strata...")
sampled_zeros = []
np.random.seed(42) # For reproducibility

for _, row in stratum_counts.iterrows():
    strat = row['stratum']
    n_target = row['target_zeros']
    
    if n_target <= 0:
        continue
        
    available_zeros = zero_df[zero_df['stratum'] == strat]
    n_available = len(available_zeros)
    
    if n_available < n_target:
        print(f"  -> Note: Not enough zeros for {strat}. Target: {n_target}, Available: {n_available}. Taking all.")
        sampled_zeros.append(available_zeros)
    else:
        sampled_zeros.append(available_zeros.sample(n=n_target, random_state=42))

final_zeros_df = pd.concat(sampled_zeros, ignore_index=True)

# ==========================================
# 4. Export
# ==========================================
final_zeros_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSuccess! Exported {len(final_zeros_df)} perfectly stratified empty images to {OUTPUT_CSV}.")
