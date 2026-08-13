# Load necessary libraries
library(tidyverse)
library(lubridate)

# Define the root directory containing the processed year folders
# =====================================================================
# 1. Locate and Combine Annotation Files (Robust Approach)
# =====================================================================
base_dir <- "../data/processed"

annotation_files <- list.files(
  path = base_dir, 
  pattern = ".*_annotations\\.csv$", 
  recursive = TRUE, 
  full.names = TRUE
)

message(sprintf("Found %d annotation files. Reading and combining...", length(annotation_files)))

# Read everything as characters to prevent type-mismatch errors during binding
raw_data <- map_dfr(
  annotation_files, 
  ~ read_csv(.x, col_types = cols(.default = col_character()), show_col_types = FALSE)
)

# =====================================================================
# 2. Clean and Condense Data (Convert Types Here)
# =====================================================================
clean_data <- raw_data %>%
  mutate(
    # Condense labels
    label = "scallop",
    
    # Convert specific columns to their proper types
    SHIP_LATITUDE = as.numeric(SHIP_LATITUDE),
    SHIP_LONGITUDE = as.numeric(SHIP_LONGITUDE),
    
    # Parse the timestamp (ymd_hms is from the lubridate package)
    IMAGE_TIMESTAMP = ymd_hms(IMAGE_TIMESTAMP),
    year = year(IMAGE_TIMESTAMP)
  )

# =====================================================================
# 3. Visualizations
# =====================================================================

# Plot 1: Total Annotations per Year
# This will help visualize the temporal balance of your massive dataset
plot_temporal <- ggplot(clean_data, aes(x = factor(year))) +
  geom_bar(fill = "steelblue", color = "black", alpha = 0.8) +
  theme_minimal() +
  labs(
    title = "Total Scallop Annotations by Year (2013 - 2024)",
    x = "Year",
    y = "Annotation Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 2: Spatial Distribution of Annotations
# Mapping SHIP_LONGITUDE and SHIP_LATITUDE to visualize survey tracks
plot_spatial <- ggplot(clean_data, aes(x = SHIP_LONGITUDE, y = SHIP_LATITUDE)) +
  # Using a low alpha (transparency) because of the sheer density of points
  geom_point(alpha = 0.05, color = "darkred", size = 0.5) +
  theme_minimal() +
  labs(
    title = "Spatial Distribution of Scallop Annotations",
    subtitle = "Aggregated survey tracks (2013-2024)",
    x = "Longitude",
    y = "Latitude"
  )

# Display the plots
print(plot_temporal)
print(plot_spatial)
