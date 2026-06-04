# ============================================================
# HabCam DS2 — audit annotated scallop images for one year
# Focus: 2022
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(readr)
  library(tibble)
  library(purrr)
  library(fs)
})

# -----------------------------
# User settings
# -----------------------------
BASE_IMG_PATH <- "/home/user/habcam_bucket/NEFSC/HabCam Survey/habcam/proc/Scall_Anno/2026tr_split/"
WRITE_IMG_PATH <- "/home/user/habcam_bucket/NEFSC/HabCam Survey/habcam/proc/Scall_Anno/2026tr_split/"
YEAR          <- 2026
zer           <- FALSE
SCALLOP_CLASS = c(185, 515, 197, 207, 920, 213, 912, 916, 525, 919, 215, 915)

# Option A: path to the year’s annotations text file (if you want script to load it)
# (adjust delimiter/reader below as needed)
ANNOT_FILE <- NULL
# ANNOT_FILE <- "/path/to/annotations_2022.txt"

# Option B: if you already have `dat` in memory, leave ANNOT_FILE as NULL.
# AnnHeader=c("annotation_id","image_id","scope_id","category_id","geometry_text","thegeom","annotator_id","assignment_id","timestamp","class_id","deprecated","geometry_id","imagename","assignment_num","percent_cover","comment","source","data_identifier")
# Ann=read.table(file=paste0("../data/raw/annotations_",YEAR,".txt"), fill=TRUE,sep="\t",na.strings=c("\\N", NA),col.names=AnnHeader,stringsAsFactors =FALSE)
# dat_scallop = subset(Ann, class_id %in% SCALLOP_CLASS)

year = as.character(YEAR)

if (zer) dat_scallop = read.csv(paste0("../data/raw/", year, "_zero.csv")) else {
  dat_scallop = read.csv(paste0("../data/raw/", year, "_annotations.csv")) %>%
    rename(imagename = image)
}

# -----------------------------
# Helpers
# -----------------------------

# Parse imagename like:
# "202203.20220530.205541302.129435.png"
# tokens: [1]=cruise? [2]=YYYYMMDD [3]=HHMMSSmmm [4]=frame? [5]=png
parse_imagename <- function(imagename) {
  # strip any accidental whitespace
  imagename <- str_trim(imagename)
  
  # If a URL or path sneaks in, keep only basename
  imagename <- basename(imagename)
  
  parts <- str_split(imagename, "\\.", simplify = TRUE)
  if (ncol(parts) < 4) {
    return(tibble(
      imagename = imagename,
      date_yyyymmdd = NA_character_,
      time_token = NA_character_,
      hour = NA_character_,
      minute = NA_character_,
      tenmin = NA_character_
    ))
  }
  
  date_yyyymmdd <- parts[, 2]
  time_token    <- parts[, 3]
  
  # time_token examples:
  # "001000052" -> 00:10:00.052 (hour=00 minute=10)
  # "205541302" -> 20:55:41.302 (hour=20 minute=55)
  hour   <- str_sub(time_token, 1, 2)
  minute <- str_sub(time_token, 3, 4)
  
  # ten-minute bin folder: floor(minute/10)*10, padded to 2 digits
  tenmin_num <- suppressWarnings(as.integer(minute) %/% 10L * 10L)
  tenmin <- ifelse(is.na(tenmin_num), NA_character_, str_pad(tenmin_num, 2, pad = "0"))
  
  tibble(
    imagename      = imagename,
    date_yyyymmdd  = date_yyyymmdd,
    time_token     = time_token,
    hour           = hour,
    minute         = minute,
    tenmin         = tenmin
  )
}

# Build expected path from folder structure:
# /Images/<YEAR>/<YYYYMM>/<YYYYMMDD>/<YYYYMMDD>_<HH>/<YYYYMMDD>_<HH><TENMIN>/<imagename>
build_expected_path <- function(imagename, base_path = BASE_IMG_PATH) {
  p <- parse_imagename(imagename)
  
  # derive year/month/day
  yyyy  <- str_sub(p$date_yyyymmdd, 1, 4)
  yyyymm <- str_sub(p$date_yyyymmdd, 1, 6)
  
  # folders
  f_year   <- yyyy
  f_month  <- yyyymm
  f_day    <- p$date_yyyymmdd
  f_hour   <- paste0(p$date_yyyymmdd, "_", p$hour)
  f_tenmin <- paste0(p$date_yyyymmdd, "_", p$hour, p$tenmin)
  
  file.path(base_path, f_year, f_month, f_day, f_hour, f_tenmin, p$imagename)
}

# Small helper to safely read file info
safe_file_info <- function(paths) {
  # returns tibble with exists/size/mtime
  exists <- file.exists(paths)
  info <- suppressWarnings(file.info(paths))
  tibble(
    exists = exists,
    size_bytes = ifelse(exists, info$size, NA_real_),
    mtime = ifelse(exists, as.character(info$mtime), NA_character_)
  )
}

# Unique image list (annotations can repeat per-image)
img_list <- dat_scallop %>%
  distinct(imagename) %>%
  filter(!is.na(imagename), imagename != "")

# -----------------------------
# Build expected paths + audit
# -----------------------------
audit <- img_list %>%
  mutate(
    expected_path = map_chr(imagename, build_expected_path, base_path = BASE_IMG_PATH)
  ) %>%
  mutate(
    write_path = map_chr(imagename, build_expected_path, base_path = WRITE_IMG_PATH)
  ) %>%
  bind_cols(safe_file_info(.$expected_path)) %>%
  mutate(
    year_from_name = suppressWarnings(as.integer(str_sub(parse_imagename(imagename)$date_yyyymmdd, 1, 4))),
    month_from_name = str_sub(parse_imagename(imagename)$date_yyyymmdd, 1, 6),
    day_from_name = parse_imagename(imagename)$date_yyyymmdd
  )

# Sanity check: are we really only looking at YEAR?
audit_year <- audit %>% filter(year_from_name == YEAR | is.na(year_from_name))

# -----------------------------
# Summary outputs
# -----------------------------
summary_overall <- audit_year %>%
  summarise(
    year = YEAR,
    scallop_images_unique = n(),
    found = sum(exists, na.rm = TRUE),
    missing = sum(!exists, na.rm = TRUE),
    pct_found = round(100 * found / scallop_images_unique, 2)
  )

summary_by_month <- audit_year %>%
  group_by(month_from_name) %>%
  summarise(
    n_images = n(),
    found = sum(exists, na.rm = TRUE),
    missing = sum(!exists, na.rm = TRUE),
    pct_found = round(100 * found / n_images, 2),
    .groups = "drop"
  ) %>%
  arrange(month_from_name)

missing_tbl <- audit_year %>%
  filter(!exists | is.na(exists)) %>%
  select(imagename, expected_path, month_from_name, day_from_name) %>%
  arrange(month_from_name, day_from_name, imagename)

found_tbl <- audit_year %>%
  filter(exists) %>%
  select(imagename, expected_path, size_bytes, mtime, month_from_name, day_from_name) %>%
  arrange(month_from_name, day_from_name, imagename)

# -----------------------------
# Print audit to console
# -----------------------------
print(summary_overall)
print(summary_by_month, n = Inf)

cat("\nMissing examples (up to 20):\n")
print(head(missing_tbl, 20))

# -----------------------------
# Write files for sharing/debugging
# -----------------------------

DEST_DIR <- "../data/raw"

if (!zer) OUT_FILE <- paste0("sources_", year, "tr.txt") else {
  OUT_FILE <- paste0("sources_", year, "_zero.txt")
}

manifest <- file.path(DEST_DIR, OUT_FILE)

# Write source paths only (1 line per image)
audit_year %>%
  filter(exists) %>%
  distinct(write_path) %>%
  pull(write_path) %>%
  writeLines(manifest)

cat("Wrote manifest:", manifest, "\n")
