import pandas as pd
from pathlib import Path

# --- Configuration ---
YEARS = [2022, 2023, 2024, 2026]
MASTER_OUT_CSV = "groundtruth2226.csv"

data_frames = []

for year in YEARS:
    file_path = Path(f"{year}/{year}_annotations.csv")
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['year'] = year
        data_frames.append(df)
    else:
        print(f"Warning: {file_path} not found. Skipping.")

if not data_frames:
    raise FileNotFoundError("No annotation files found.")

# ==========================================
# 1. Combine into one master dataframe
# ==========================================
master_df = pd.concat(data_frames, ignore_index=True)

# Normalize column names to lowercase to make filtering robust
master_df.columns = [c.lower() for c in master_df.columns]

# Map known column variants to standard names used in the pipeline
rename_map = {
    'ship_latitude': 'latitude',
    'ship_longitude': 'longitude',
    'image': 'imagename',
    'altimeter_altitude_meter': 'altitude'
}
master_df.rename(columns=rename_map, inplace=True)

initial_count = len(master_df)
initial_images = master_df['imagename'].nunique()

# ==========================================
# 2. Sanitize Data
# ==========================================
# Remove NAs for spatial coordinates
clean_df = master_df.dropna(subset=['latitude', 'longitude']).copy()

# Filter spatial anomalies: Longitude must be between -76 and -66
clean_df = clean_df[(clean_df['longitude'] >= -76) & (clean_df['longitude'] <= -66)]

# Filter altitude > 3 (poor image quality safeguard)
if 'altitude' in clean_df.columns:
    clean_df = clean_df[clean_df['altitude'] <= 3]
else:
    print("Note: 'altitude' column not found in data. Skipping altitude filter.")

final_count = len(clean_df)
final_images = clean_df['imagename'].nunique()

print(f"\n--- Data Sanitization Summary ---")
print(f"Initial annotations: {initial_count} (across {initial_images} unique images)")
print(f"Final clean annotations: {final_count} (across {final_images} unique images)")
print(f"Removed {initial_count - final_count} anomalous/missing records.")

# ==========================================
# 3. Export and Recalculate Stratification
# ==========================================
# Save the sanitized master dataframe
clean_df.to_csv(MASTER_OUT_CSV, index=False)
print(f"\nSaved clean master annotations to {MASTER_OUT_CSV}")

# Filter down to just the target years for empty-sourcing (2022-2024)
pos_22_24 = clean_df[clean_df['year'].isin([2022, 2023, 2024])]
unique_pos_images_22_24 = pos_22_24['imagename'].nunique()

# Recalculate the 15% target
target_empty_count = int(final_images * 0.15)
print(f"New target empty images to source (15% of 2022-2026): {target_empty_count}")

# Recalculate Bin Edges based on the CLEANED image footprints
image_spatial_meta = pos_22_24.drop_duplicates(subset=['imagename'])
lat_bins, lat_edges = pd.qcut(image_spatial_meta['latitude'], q=4, retbins=True, duplicates="drop")
lon_bins, lon_edges = pd.qcut(image_spatial_meta['longitude'], q=4, retbins=True, duplicates="drop")

print(f"\nNew Latitude Bin Edges: {lat_edges}")
print(f"New Longitude Bin Edges: {lon_edges}")