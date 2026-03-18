suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(ggplot2)
  library(magick)
})

# ============================================================
# Settings
# ============================================================
# IMG_DIR_FULL   <- "../data/images/2022tr"          # full stereo images (flat)
# IMG_DIR_FULL   <- "/mnt/s/saltnoaa/images/2022tr"          
IMG_DIR_FULL <- "/Volumes/PortableSSD/saltnoaa/images/2022/"
IMG_DIR_LEFT   <- "../data/images/2022"     # output left-only images
OUT_DIR        <- "../data/audit_2022tr"           # audit outputs
N_MONTAGE      <- 6
# SET_SEED       <- 7
OVERWRITE_LEFT <- FALSE
YEAR = 2022
SCALLOP_CLASS = c(185, 515, 197, 207, 920, 213, 912, 916, 525, 919, 215, 915)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(IMG_DIR_LEFT, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# line segment -> bounding box (your function)
# ============================================================
lineseg2bb <- function(x1, y1, x2, y2) {
  radius  <- sqrt((y2 - y1)^2 + (x2 - x1)^2) / 2
  centerx <- mean(c(x1, x2))
  centery <- mean(c(y1, y2))
  tlx <- centerx - radius
  tly <- centery - radius
  brx <- centerx + radius
  bry <- centery + radius
  c(tlx, tly, brx, bry)
}

# ============================================================
# Geometry typing + parsing
# ============================================================

geom_type_from_text <- function(txt) {
  if (is.na(txt)) return("empty")
  txt <- str_trim(txt)
  if (txt == "" | tolower(txt) %in% c("na", "null")) return("empty")
  
  # common patterns
  if (str_detect(txt, "^\\s*line\\s*:"))  return("line")
  if (str_detect(txt, "^\\s*point\\s*:")) return("point")
  
  # if it's got 4+ numbers it might be a bbox/polygon/etc.
  n_nums <- length(str_extract_all(txt, "[-+]?[0-9]*\\.?[0-9]+")[[1]])
  if (n_nums == 0) return("empty")
  if (n_nums == 2) return("point")  # often point has two numbers
  if (n_nums >= 4) return("other")
  
  "other"
}

parse_first_n_numbers <- function(txt, n = 4) {
  nums <- str_extract_all(txt, "[-+]?[0-9]*\\.?[0-9]+")[[1]]
  if (length(nums) < n) return(rep(NA_real_, n))
  as.numeric(nums[1:n])
}

# For line: we need 4 numbers (x1,y1,x2,y2)
parse_line_coords <- function(geometry_text) {
  parse_first_n_numbers(geometry_text, n = 4)
}

# ============================================================
# Bounding box clipping to left image bounds
# ============================================================
clip_bbox_to_bounds <- function(tlx, tly, brx, bry, width, height) {
  tlx2 <- pmax(0, pmin(tlx, width  - 1))
  brx2 <- pmax(0, pmin(brx, width  - 1))
  tly2 <- pmax(0, pmin(tly, height - 1))
  bry2 <- pmax(0, pmin(bry, height - 1))
  c(tlx2, tly2, brx2, bry2)
}

remap_bbox_to_left_half <- function(tlx, tly, brx, bry, full_w, full_h) {
  half_w <- floor(full_w / 2)
  
  # entirely right side => drop
  if (is.na(tlx) || is.na(brx) || tlx >= half_w) {
    return(c(NA_real_, NA_real_, NA_real_, NA_real_, half_w, full_h, "dropped_right"))
  }
  
  # clip horizontally to left-half bounds [0, half_w)
  tlx_c <- tlx
  brx_c <- pmin(brx, half_w - 1)
  
  # also clip vertically to [0, full_h)
  clipped <- clip_bbox_to_bounds(tlx_c, tly, brx_c, bry, width = half_w, height = full_h)
  
  status <- ifelse(brx > (half_w - 1), "clipped_split", "ok")
  c(clipped[1], clipped[2], clipped[3], clipped[4], half_w, full_h, status)
}

# ============================================================
# Split image (stereo) -> left half and save
# ============================================================
split_and_save_left_safe <- function(src_path, dest_path, overwrite = FALSE, retry = 1) {
  if (!file.exists(src_path)) {
    return(list(ok = FALSE, status = "missing_src", msg = "source file not found"))
  }
  if (file.exists(dest_path) && !overwrite) {
    return(list(ok = TRUE, status = "already_exists", msg = NA_character_))
  }
  
  attempt <- 0
  last_err <- NA_character_
  
  while (attempt <= retry) {
    attempt <- attempt + 1
    res <- tryCatch({
      im <- image_read(src_path)
      info <- image_info(im)
      full_w <- info$width[1]
      full_h <- info$height[1]
      half_w <- floor(full_w / 2)
      
      left <- image_crop(im, geometry = paste0(half_w, "x", full_h, "+0+0"))
      image_write(left, path = dest_path, format = "png")
      
      list(ok = TRUE, status = "saved", msg = NA_character_)
    }, error = function(e) {
      last_err <<- conditionMessage(e)
      NULL
    })
    
    if (!is.null(res)) return(res)
    
    # brief pause can help with flaky /mnt/* reads
    Sys.sleep(0.05)
  }
  
  list(ok = FALSE, status = "read_error", msg = last_err)
}


# ============================================================
# 0) Preconditions: expect dat_scallop exists
# ============================================================
# stopifnot(exists("dat_scallop"))
SCALLOP_CLASS = c(185, 515, 197, 207, 920, 213, 912, 916, 525, 919, 215, 915)
AnnHeader=c("annotation_id","image_id","scope_id","category_id","geometry_text","thegeom","annotator_id","assignment_id","timestamp","class_id","deprecated","geometry_id","imagename","assignment_num","percent_cover","comment","source","data_identifier")
Ann=read.table(file=paste0("../data/raw/annotations_",YEAR,".txt"), fill=TRUE,sep="\t",na.strings=c("\\N", NA),col.names=AnnHeader,stringsAsFactors =FALSE)
dat_scallop = subset(Ann, class_id %in% SCALLOP_CLASS)
# Ensure imagename + paths
dat_scallop2 <- dat_scallop %>%
  mutate(
    imagename  = str_trim(imagename),
    # img_full   = file.path(IMG_DIR_FULL, imagename),
    # img_left   = file.path(IMG_DIR_LEFT, imagename),
    # img_exists = file.exists(img_full),
    geom_type  = vapply(geometry_text, geom_type_from_text, character(1))
  )

# ============================================================
# 1) Convert LINE annotations to bboxes; keep point/empty rows for audit
# ============================================================
ann <- dat_scallop2 %>%
  mutate(
    # parse coords only for line
    coords = if_else(geom_type == "line", geometry_text, NA_character_),
    coords = map(coords, ~ if (is.na(.x)) rep(NA_real_, 4) else parse_line_coords(.x)),
    x1 = map_dbl(coords, 1),
    y1 = map_dbl(coords, 2),
    x2 = map_dbl(coords, 3),
    y2 = map_dbl(coords, 4),
    
    # bbox only for lines
    bb = pmap(list(x1, y1, x2, y2), ~ if (any(is.na(c(..1, ..2, ..3, ..4)))) rep(NA_real_, 4) else lineseg2bb(..1, ..2, ..3, ..4)),
    tlx = map_dbl(bb, 1),
    tly = map_dbl(bb, 2),
    brx = map_dbl(bb, 3),
    bry = map_dbl(bb, 4),
    
    box_w = brx - tlx,
    box_h = bry - tly,
    box_area = box_w * box_h,
    
    # flags
    geom_empty = geom_type == "empty",
    geom_point = geom_type == "point",
    geom_line  = geom_type == "line",
    geom_other = geom_type == "other",
    
    bad_geom_line = geom_line & (is.na(tlx) | is.na(tly) | is.na(brx) | is.na(bry)),
    bad_box_line  = geom_line & (!bad_geom_line) & (box_w <= 0 | box_h <= 0)
  ) %>%
  select(-coords, -bb)

# Save full annotation table (includes non-line rows)
ann_file <- file.path(OUT_DIR, "annotations_2022tr_scallop_all_geoms.csv")
write_csv(ann, ann_file)

# ============================================================
# 2) Audit summaries (explicit empty/point counts)
# ============================================================
img_summary <- ann %>%
  group_by(imagename) %>%
  summarise(
    img_exists = any(img_exists),
    n_rows = n(),
    n_line = sum(geom_line),
    n_point = sum(geom_point),
    n_empty = sum(geom_empty),
    n_other = sum(geom_other),
    n_bad_line_geom = sum(bad_geom_line),
    n_bad_line_box  = sum(bad_box_line),
    annotators = paste(sort(unique(annotator_id)), collapse = ","),
    .groups = "drop"
  ) %>%
  arrange(desc(n_line), desc(n_rows))

overall_summary <- ann %>%
  summarise(
    n_rows = n(),
    n_images = n_distinct(imagename),
    n_annotators = n_distinct(annotator_id),
    pct_missing_images = round(100 * mean(!img_exists), 3),
    pct_line = round(100 * mean(geom_line), 3),
    pct_point = round(100 * mean(geom_point), 3),
    pct_empty = round(100 * mean(geom_empty), 3),
    pct_other = round(100 * mean(geom_other), 3),
    pct_bad_line_geom = round(100 * mean(bad_geom_line), 3),
    pct_bad_line_box  = round(100 * mean(bad_box_line), 3)
  )

annotator_summary <- ann %>%
  count(annotator_id, geom_type, sort = TRUE) %>%
  tidyr::pivot_wider(names_from = geom_type, values_from = n, values_fill = 0) %>%
  mutate(total = line + point + empty + other) %>%
  arrange(desc(total))

# write_csv(img_summary, file.path(OUT_DIR, "image_summary_2022tr.csv"))
# write_csv(annotator_summary, file.path(OUT_DIR, "annotator_summary_2022tr.csv"))
# write_csv(overall_summary, file.path(OUT_DIR, "overall_summary_2022tr.csv"))

print(overall_summary)

# ============================================================
# 3) Split FULL stereo images -> LEFT images (batch, low-memory)
# ============================================================
unique_imgs <- ann %>%
  distinct(imagename, img_full, img_left, img_exists) %>%
  filter(img_exists)

# ---- run it on your unique_imgs (one at a time; does not crash) ----
split_log_file <- file.path(OUT_DIR, "split_left_failures.csv")

split_results <- map(seq_len(nrow(unique_imgs)), function(i) {
  r <- split_and_save_left_safe(
    src_path  = unique_imgs$img_full[i],
    dest_path = unique_imgs$img_left[i],
    overwrite = OVERWRITE_LEFT,
    retry = 1
  )
  
  tibble(
    imagename = unique_imgs$imagename[i],
    img_full  = unique_imgs$img_full[i],
    img_left  = unique_imgs$img_left[i],
    ok        = r$ok,
    status    = r$status,
    msg       = r$msg
  )
})

split_audit <- bind_rows(split_results) %>%
  mutate(left_exists = file.exists(img_left))

# save full audit
write_csv(split_audit, file.path(OUT_DIR, "split_left_audit_2022tr.csv"))

# save failures only (easier to inspect)
failures <- split_audit %>% filter(!ok)
write_csv(failures, split_log_file)

split_audit %>%
  count(status, sort = TRUE) %>%
  print(n = Inf)

cat("\nFailures written to:\n", split_log_file, "\n")

# ============================================================
# 4) OMITTED: remap LINE bboxes to LEFT coords + clip
#    (We will instead clip bboxes on-the-fly for montage only.)
# ============================================================

# ============================================================
# 5) Montage: 6 random LEFT images with bboxes clipped to left-half
# ============================================================
# ---- precompute once (outside loop) ----
valid_imgs <- ann %>%
  filter(
    geom_type == "line",
    !is.na(tlx), !is.na(tly), !is.na(brx), !is.na(bry),
    file.exists(file.path(IMG_DIR_LEFT, imagename))
  ) %>%
  distinct(imagename) %>%
  pull(imagename)

if (length(valid_imgs) == 0) stop("No valid images found for montage (line bboxes + left image exists).")

clip_bbox_to_left_image <- function(tlx, tly, brx, bry, left_w, left_h) {
  if (is.na(tlx) || is.na(brx) || tlx >= left_w) return(NULL)
  
  tlx2 <- max(0, min(tlx, left_w - 1))
  brx2 <- max(0, min(brx, left_w - 1))
  tly2 <- max(0, min(tly, left_h - 1))
  bry2 <- max(0, min(bry, left_h - 1))
  
  if (brx2 <= tlx2 || bry2 <= tly2) return(NULL)
  
  tibble(
    tlx = tlx2, tly = tly2, brx = brx2, bry = bry2,
    clipped = (brx > (left_w - 1))
  )
}

draw_boxes_one_left_clipped <- function(imagename, ann_df, img_dir_left) {
  left_path <- file.path(img_dir_left, imagename)
  
  im_left <- image_read(left_path)
  info <- image_info(im_left)
  left_w <- info$width[1]
  left_h <- info$height[1]
  
  this <- ann_df %>%
    filter(imagename == .env$imagename, geom_type == "line") %>%
    select(tlx, tly, brx, bry)
  
  clipped_df <- pmap_dfr(
    list(this$tlx, this$tly, this$brx, this$bry),
    ~ clip_bbox_to_left_image(..1, ..2, ..3, ..4, left_w = left_w, left_h = left_h)
  )
  
  n_total <- nrow(this)
  n_kept  <- nrow(clipped_df)
  n_clip  <- ifelse(n_kept > 0, sum(clipped_df$clipped), 0)
  
  # ---- KEY CHANGE: draw ONCE per image (not once per box) ----
  if (n_kept > 0) {
    im_left <- image_draw(im_left)
    rect(
      xleft   = clipped_df$tlx,
      ytop    = clipped_df$tly,
      xright  = clipped_df$brx,
      ybottom = clipped_df$bry,
      border  = "red",
      lwd     = 3
    )
    dev.off()
  }
  
  image_annotate(
    im_left,
    text = paste0(imagename, " | line=", n_total, " kept=", n_kept, " clipped=", n_clip),
    gravity = "northwest",
    size = 28,
    color = "white",
    boxcolor = "black"
  )
}

# --------------------------
# loop montages with different seeds
# --------------------------
make_montage_for_seed <- function(seed, n_montage = 6) {
  set.seed(seed)
  
  pick_imgs <- sample(valid_imgs, size = min(n_montage, length(valid_imgs)), replace = FALSE)
  
  annotated_imgs <- map(
    pick_imgs,
    ~ tryCatch(
      draw_boxes_one_left_clipped(.x, ann_df = ann, img_dir_left = IMG_DIR_LEFT),
      error = function(e) { message("FAIL ", .x, " (seed=", seed, "): ", conditionMessage(e)); NULL }
    )
  ) %>% compact()
  
  if (length(annotated_imgs) == 0) {
    message("Seed ", seed, ": no annotated images produced.")
    return(invisible(NULL))
  }
  
  n <- length(annotated_imgs)
  tile_x <- ceiling(sqrt(n))
  tile_y <- ceiling(n / tile_x)
  
  montage <- image_montage(
    do.call(c, annotated_imgs),
    tile = paste0(tile_x, "x", tile_y),
    geometry = "+2+2"
  )
  
  montage_file <- file.path(OUT_DIR, paste0("montage_2022tr_left_seed", seed, ".png"))
  image_write(montage, path = montage_file, format = "png")
  
  # append log
  log_file <- file.path(OUT_DIR, "montage_images_used.txt")
  write_lines(c(paste0("seed=", seed), pick_imgs, ""), log_file, append = TRUE)
  
  rm(annotated_imgs, montage)
  gc()
  
  cat("Saved montage:", montage_file, "\n")
  invisible(montage_file)
}

# Example: seeds 1–20
for (seed in 1:20) {
  make_montage_for_seed(seed, n_montage = N_MONTAGE)
}
