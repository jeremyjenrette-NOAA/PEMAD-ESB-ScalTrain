library(dplyr)
library(purrr)

YEAR = 2026

spread = paste0("./", YEAR, "/", YEAR, "_annotations.csv")

# 1. Load the data
dat <- read.csv("groundtruth2226_supplemental.csv")

print(paste0(nrow(dat)," Annotations"))

# 2. Define a function to calculate the Intersection over Union (IoU)
calculate_iou <- function(box1, box2) {
  # Format expected: c(TLx, TLy, BRx, BRy)
  
  # Find coordinates of the intersection rectangle
  x_left <- max(box1[1], box2[1])
  y_top <- max(box1[2], box2[2])
  x_right <- min(box1[3], box2[3])
  y_bottom <- min(box1[4], box2[4])
  
  # If the boxes don't overlap, x_right will be less than x_left (or y_bottom < y_top)
  if (x_right <= x_left || y_bottom <= y_top) {
    return(0.0)
  }
  
  # Calculate areas
  intersection_area <- (x_right - x_left) * (y_bottom - y_top)
  box1_area <- (box1[3] - box1[1]) * (box1[4] - box1[2])
  box2_area <- (box2[3] - box2[1]) * (box2[4] - box2[2])
  
  # Calculate IoU
  iou <- intersection_area / (box1_area + box2_area - intersection_area)
  return(iou)
}

# 3. Define a function to evaluate overlapping boxes within a single image
remove_duplicates <- function(img_data, threshold = 0.92) {
  # If there's only one annotation (or none), return as is
  if (nrow(img_data) <= 1) return(img_data)
  
  keep_indices <- c()
  
  for (i in 1:nrow(img_data)) {
    # Extract coordinates for the current row
    box_i <- as.numeric(img_data[i, c("tlx", "tly", "brx", "bry")])
    is_duplicate <- FALSE
    
    # Compare against boxes we've already decided to keep
    if (length(keep_indices) > 0) {
      for (j in keep_indices) {
        box_j <- as.numeric(img_data[j, c("tlx", "tly", "brx", "bry")])
        
        # Check overlap
        iou <- calculate_iou(box_i, box_j)
        
        if (iou >= threshold) {
          is_duplicate <- TRUE
          break # Stop checking; we already know it's a duplicate
        }
      }
    }
    
    # If it didn't overlap >= 92% with any kept boxes, add it to the keep list
    if (!is_duplicate) {
      keep_indices <- c(keep_indices, i)
    }
  }
  
  # Return only the rows we flagged to keep
  return(img_data[keep_indices, ])
}

# 4. Apply the deduplication across the entire dataset
clean_dat <- dat %>%
  group_split(imagename) %>% # Split the dataframe into a list of dataframes by image
  map_dfr(~ remove_duplicates(.x, threshold = 0.95)) # Apply function and bind rows back together

print(paste0(nrow(clean_dat)," Annotations"))

# 5. Review and save your clean data
head(clean_dat)
write.csv(clean_dat, file = "./supp.csv", row.names = FALSE, na = "")

