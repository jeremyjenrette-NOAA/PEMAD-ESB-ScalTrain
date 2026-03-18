library(data.table)
library(stringr)
library(fs)

# -------------------------
# Paths
# -------------------------
img_dir <- "/Volumes/PortableSSD/saltnoaa/images/2024/"
anno_csv <- "../data/raw/groundtruth24.csv"
anno_right_shifted_path <- "/Volumes/PortableSSD/saltnoaa/images/2024_right/groundtruth24.csv"
out_img_dir <- "/Volumes/PortableSSD/saltnoaa/images/2024_right/image_neg"

dir_create(out_img_dir)

# -------------------------
# Load data
# -------------------------
anno <- as.data.table(read.csv(anno_csv))
anno_shifted <- as.data.table(read.csv(anno_right_shifted_path))

# -------------------------
# Step 1: Get all images
# -------------------------
img_files <- list.files(
  img_dir,
  pattern = "\\.(png|jpg|jpeg|tif|tiff)$",
  full.names = FALSE,
  ignore.case = TRUE
)

# -------------------------
# Step 2: Identify scallop images
# -------------------------
anno[, ClassName := tolower(trimws(ClassName))]

scallop_imgs <- unique(anno[ClassName == "scallop", Imagename])

# -------------------------
# Step 3: Find NEGATIVE images
# -------------------------
neg_imgs <- setdiff(img_files, scallop_imgs)[1:1000]

cat("Total images:", length(img_files), "\n")
cat("Scallop images:", length(scallop_imgs), "\n")
cat("Negative images:", length(neg_imgs), "\n")

# -------------------------
# Step 4: Create negative annotation table
# -------------------------
neg_dt <- data.table(
  image = neg_imgs,
  TLx = NA_real_,
  TLy = NA_real_,
  BRx = NA_real_,
  BRy = NA_real_,
  label = "negative",
  date = as.Date(NA),
  shift_applied = FALSE
)
# neg_dt = neg_dt[1:1000]
# -------------------------
# Step 5: Combine with shifted annotations
# -------------------------
anno_final <- rbindlist(
  list(anno_shifted, neg_dt),
  fill = TRUE
)

# -------------------------
# Step 6: Copy negative images
# -------------------------
cat("\nCopying negative images...\n")

file_copy(
  path = file.path(img_dir, neg_imgs),
  new_path = file.path(out_img_dir, neg_imgs),
  overwrite = FALSE
)

cat("Done copying negatives.\n")

# -------------------------
# Step 7: Save final dataset
# -------------------------
out_csv <- "/Volumes/PortableSSD/saltnoaa/images/2024_right/groundtruth24neg.csv"

fwrite(anno_final, out_csv)

cat("\nSaved final dataset →", out_csv, "\n")
