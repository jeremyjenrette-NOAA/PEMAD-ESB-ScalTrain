dat = read.csv("2026/2026_annotations.csv")

# 1. Define the combined calculation function
calc_camera_metrics <- function(altitude, roll, pitch) {
  # Return NAs if any required sensor data is missing
  if (is.na(altitude) || is.na(roll) || is.na(pitch)) {
    return(c(FOV = NA, mm_px = NA))
  }
  
  # Constants
  DTOR <- pi / 180
  focalLength <- 16 * 0.00133
  PIXEL_SIZE <- 0.00000586
  TOTAL_PIXELS <- 1936 * 1216
  
  # Trig
  sP <- sin(pitch * DTOR); cP <- cos(pitch * DTOR)
  sR <- sin(roll  * DTOR); cR <- cos(roll  * DTOR)
  
  # Rotation matrix
  m <- matrix(0, nrow = 3, ncol = 3)
  m[1,1] <- cP;        m[1,2] <- 0;   m[1,3] <- -sP
  m[2,1] <- sP*sR;     m[2,2] <- cR;  m[2,3] <- cP*sR
  m[3,1] <- sP*cR;     m[3,2] <- -sR; m[3,3] <- cP*cR
  
  # Image corners in sensor coords
  ulX <- -1936/2 * PIXEL_SIZE; ulY <-  1216/2 * PIXEL_SIZE
  urX <- -ulX;                 urY <-  ulY
  llX <-  ulX;                 llY <- -ulY
  lrX <- -ulX;                 lrY <- -ulY
  
  # Store corners in order (ul, ll, lr, ur)
  px <- c(ulX, llX, lrX, urX)
  py <- c(ulY, llY, lrY, urY)
  
  # Project rays through rotation matrix
  X <- numeric(4); Y <- numeric(4); Z <- numeric(4)
  for (k in 1:4) {
    X[k] <- m[1,1]*px[k] + m[1,2]*py[k] + m[1,3]*(-focalLength)
    Y[k] <- m[2,1]*px[k] + m[2,2]*py[k] + m[2,3]*(-focalLength)
    Z[k] <- m[3,1]*px[k] + m[3,2]*py[k] + m[3,3]*(-focalLength)
  }
  
  # Prevent division by zero or positive Z values (pointing above horizon)
  if (any(Z >= 0)) {
    return(c(FOV = NA, mm_px = NA))
  }
  
  # Intersect with seabed plane at given altitude
  X <- X * (altitude / Z)
  Y <- Y * (altitude / Z)
  
  # Polygon area (shoelace)
  area <- 0
  j <- 4
  for (i in 1:4) {
    area <- area + (X[j] + X[i]) * (Y[j] - Y[i])
    j <- i
  }
  
  # Calculations
  fov_sq_m <- abs(area / 2) # Use absolute value to guarantee positive area
  avg_mm_px <- sqrt(fov_sq_m / TOTAL_PIXELS) * 1000
  
  return(c(FOV = fov_sq_m, mm_px = avg_mm_px))
}


# 2. Master Population Routine

# Check and calculate BOTTOM_DEPTH
if (!"BOTTOM_DEPTH" %in% names(dat) || any(is.na(dat$BOTTOM_DEPTH))) {
  # Calculate only where it's missing to avoid overwriting existing valid data
  needs_depth <- is.na(dat$BOTTOM_DEPTH) | !("BOTTOM_DEPTH" %in% names(dat))
  
  # If column doesn't exist at all, initialize it with NAs
  if (!"BOTTOM_DEPTH" %in% names(dat)) {
    dat$BOTTOM_DEPTH <- NA 
  }
  
  dat$BOTTOM_DEPTH <- ifelse(
    is.na(dat$BOTTOM_DEPTH), 
    dat$ALTIMETER_ALTITUDE_METER + dat$CTD_VEHICLE_DEPTH_METER, 
    dat$BOTTOM_DEPTH
  )
}

# Check and calculate FOV and MM_PER_PIXEL
# Initialize columns if they are completely missing from the dataframe
if (!"FIELD_OF_VIEW_SQ_METER" %in% names(dat)) dat$FIELD_OF_VIEW_SQ_METER <- NA
if (!"MILLIMETER_PER_PIXEL" %in% names(dat)) dat$MILLIMETER_PER_PIXEL <- NA

# Identify rows that need computation
needs_fov <- is.na(dat$FIELD_OF_VIEW_SQ_METER)
needs_mm_px <- is.na(dat$MILLIMETER_PER_PIXEL)
rows_to_process <- needs_fov | needs_mm_px

if (any(rows_to_process)) {
  # Run calculations for the required rows using mapply
  results <- t(mapply(
    calc_camera_metrics,
    altitude = dat$ALTIMETER_ALTITUDE_METER[rows_to_process],
    roll = dat$VEHICLE_ROLL_ANGLE[rows_to_process],
    pitch = dat$VEHICLE_PITCH_ANGLE[rows_to_process]
  ))
  
  # Update dataframe
  dat$FIELD_OF_VIEW_SQ_METER[needs_fov] <- results[needs_fov[rows_to_process], "FOV"]
  dat$MILLIMETER_PER_PIXEL[needs_mm_px] <- results[needs_mm_px[rows_to_process], "mm_px"]
}

# dat$CRUISE_ID = 202605
dat$CLASS_NAME <- dat$label
dat$label <- "scallop"

write.csv(dat, "./2026_annotations.csv", row.names = FALSE)