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
IMG_DIR_FULL   <- "../data/images/2022tr"          # full stereo images (flat)
IMG_DIR_LEFT   <- "../data/images/2022tr_split"     # output left-only images
OUT_DIR        <- "../data/audit_2022tr"           # audit outputs
N_MONTAGE      <- 6
SET_SEED       <- 7
OVERWRITE_LEFT <- FALSE
YEAR = 2022
SCALLOP_CLASS = 185

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
split_and_save_left <- function(src_path, dest_path, overwrite = FALSE) {
  if (!file.exists(src_path)) return(FALSE)
  if (file.exists(dest_path) && !overwrite) return(TRUE)
  
  im <- image_read(src_path)
  info <- image_info(im)
  
  full_w <- info$width[1]
  full_h <- info$height[1]
  half_w <- floor(full_w / 2)
  
  # crop left half: geometry = "<w>x<h>+x+y"
  left <- image_crop(im, geometry = paste0(half_w, "x", full_h, "+0+0"))
  image_write(left, path = dest_path, format = "png")
  TRUE
}

# ============================================================
# 0) Preconditions: expect dat_scallop exists
# ============================================================
# stopifnot(exists("dat_scallop"))
AnnHeader=c("annotation_id","image_id","scope_id","category_id","geometry_text","thegeom","annotator_id","assignment_id","timestamp","class_id","deprecated","geometry_id","imagename","assignment_num","percent_cover","comment","source","data_identifier")
Ann=read.table(file=paste0("../data/raw/annotations_",YEAR,".txt"), fill=TRUE,sep="\t",na.strings=c("\\N", NA),col.names=AnnHeader,stringsAsFactors =FALSE)
dat_scallop = subset(Ann, class_id == SCALLOP_CLASS)
# Ensure imagename + paths
dat_scallop2 <- dat_scallop %>%
  mutate(
    imagename  = str_trim(imagename),
    img_full   = file.path(IMG_DIR_FULL, imagename),
    img_left   = file.path(IMG_DIR_LEFT, imagename),
    img_exists = file.exists(img_full),
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

# If you have tons, this still stays light because it processes one at a time.
split_ok <- map_lgl(seq_len(nrow(unique_imgs)), function(i) {
  split_and_save_left(unique_imgs$img_full[i], unique_imgs$img_left[i], overwrite = OVERWRITE_LEFT)
})

split_audit <- unique_imgs %>%
  mutate(left_saved = split_ok,
         left_exists = file.exists(img_left))

write_csv(split_audit, file.path(OUT_DIR, "split_left_audit_2022tr.csv"))

# ============================================================
# 4) Remap LINE bboxes to LEFT image coordinates + clip
# ============================================================
# We need image widths/heights to remap properly. We'll read info per image once.
img_dims <- split_audit %>%
  filter(left_exists) %>%
  select(imagename, img_full, img_left) %>%
  mutate(
    # read info from FULL image (since original coords are in full stereo space)
    info = map(img_full, ~ image_info(image_read(.x))),
    full_w = map_dbl(info, ~ .x$width[1]),
    full_h = map_dbl(info, ~ .x$height[1])
  ) %>%
  select(-info)

# Join dims onto annotations
ann2 <- ann %>%
  left_join(img_dims, by = "imagename")

# Remap only line bboxes
remapped <- ann2 %>%
  mutate(
    remap = pmap(
      list(tlx, tly, brx, bry, full_w, full_h, geom_type),
      function(tlx, tly, brx, bry, full_w, full_h, geom_type) {
        if (geom_type != "line" || any(is.na(c(tlx, tly, brx, bry, full_w, full_h)))) {
          return(c(NA_real_, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_, "not_line_or_missing_dims"))
        }
        remap_bbox_to_left_half(tlx, tly, brx, bry, full_w, full_h)
      }
    ),
    tlx_left = map_dbl(remap, 1),
    tly_left = map_dbl(remap, 2),
    brx_left = map_dbl(remap, 3),
    bry_left = map_dbl(remap, 4),
    left_w   = map_dbl(remap, 5),
    left_h   = map_dbl(remap, 6),
    split_status = map_chr(remap, 7),
    
    # recompute box metrics in left coords
    box_w_left = brx_left - tlx_left,
    box_h_left = bry_left - tly_left,
    box_area_left = box_w_left * box_h_left,
    
    dropped_in_left = split_status == "dropped_right" | is.na(tlx_left) | is.na(brx_left)
  ) %>%
  select(-remap)

# Save “left-ready” annotations
ann_left_file <- file.path(OUT_DIR, "annotations_2022tr_left_bboxes.csv")
write_csv(remapped, ann_left_file)
cat("Wrote left-remapped annotations to:\n", ann_left_file, "\n")

# Quick summary of split effects
split_effects <- remapped %>%
  filter(geom_type == "line") %>%
  count(split_status, sort = TRUE)

write_csv(split_effects, file.path(OUT_DIR, "split_effects_summary.csv"))
print(split_effects)

# ============================================================
# 5) Montage: 6 random LEFT images with LEFT bboxes
# ============================================================
set.seed(SET_SEED)

valid_imgs <- remapped %>%
  filter(
    geom_type == "line",
    !dropped_in_left,
    !is.na(tlx_left), !is.na(tly_left), !is.na(brx_left), !is.na(bry_left),
    file.exists(file.path(IMG_DIR_LEFT, imagename))
  ) %>%
  group_by(imagename) %>%
  summarise(n_ann = n(), .groups = "drop") %>%
  filter(n_ann > 0) %>%
  pull(imagename) %>%
  unique()

if (length(valid_imgs) == 0) stop("No valid images found for montage after remap.")

pick_imgs <- sample(valid_imgs, size = min(N_MONTAGE, length(valid_imgs)), replace = FALSE)

draw_boxes_one_left <- function(imagename, ann_df, img_dir) {
  this <- ann_df %>%
    filter(imagename == .env$imagename, geom_type == "line", !dropped_in_left)
  
  path <- file.path(img_dir, imagename)
  im <- image_read(path)
  
  for (i in seq_len(nrow(this))) {
    im <- image_draw(im)
    rect(
      xleft   = this$tlx_left[i],
      ytop    = this$tly_left[i],
      xright  = this$brx_left[i],
      ybottom = this$bry_left[i],
      border  = "red",
      lwd     = 3
    )
    dev.off()
  }
  
  image_annotate(
    im,
    text = paste0(imagename, " (n=", nrow(this), ", ", paste0(unique(this$split_status), collapse = ","), ")"),
    gravity = "northwest",
    size = 28,
    color = "white",
    boxcolor = "black"
  )
}

annotated_imgs <- map(pick_imgs, draw_boxes_one_left, ann_df = remapped, img_dir = IMG_DIR_LEFT)

# montage layout: 2x3 for 6, otherwise roughly square
n <- length(annotated_imgs)
tile_x <- ceiling(sqrt(n))
tile_y <- ceiling(n / tile_x)

montage <- image_montage(
  do.call(c, annotated_imgs),
  tile = paste0(tile_x, "x", tile_y),
  geometry = "+2+2"
)

montage_file <- file.path(OUT_DIR, "montage_2022tr_left_random.png")
image_write(montage, path = montage_file, format = "png")
write_lines(pick_imgs, file.path(OUT_DIR, "montage_images_used.txt"))

cat("\nSaved montage:\n", montage_file, "\n")
