# ============================================================
# VIAME groundtruth prep: clip bboxes, QA/QC plot, export
# ============================================================

library(dplyr)
library(readr)
library(stringr)
library(magick)

# ---- USER EDITS ----
CSV_IN  <- "../data/processed/groundtruth2022_line.csv"   # your current CSV
IMG_DIR <- "/path/to/2022/images"       # folder containing the .png images
OUT_DIR <- "../data/processed/" # output training root
SEQ_DIR <- file.path(OUT_DIR, "seq1")   # VIAME expects a seq folder
set.seed(7)                             # for reproducible random QA/QC pick

# ---- 1) Read your input CSV (keeps your cols; we’ll select what we need) ----
gt_raw <- read_csv(CSV_IN, show_col_types = FALSE)

# Normalize column names if yours are quoted or slightly different
# (adjust here if needed)
gt <- gt_raw %>%
  rename(
    image = image,
    frame = frame,
    TLx   = TLx,
    TLy   = TLy,
    BRx   = BRx,
    BRy   = BRy,
    label = label
  ) %>%
  mutate(
    image = as.character(image),
    label = as.character(label),
    frame = as.integer(frame),
    TLx = as.numeric(TLx), TLy = as.numeric(TLy),
    BRx = as.numeric(BRx), BRy = as.numeric(BRy)
  )

# ---- 2) Get true image dimensions for each unique image ----
# This reads headers via ImageMagick; fast enough for many images.
img_dims <- gt %>%
  distinct(image) %>%
  mutate(
    path = file.path(IMG_DIR, image),
    exists = file.exists(path)
  )

if (any(!img_dims$exists)) {
  missing <- img_dims %>% filter(!exists) %>% pull(image)
  warning("Some images referenced in CSV do not exist in IMG_DIR. Example(s):\n",
          paste(head(missing, 10), collapse = "\n"))
}

img_dims <- img_dims %>%
  filter(exists) %>%
  rowwise() %>%
  mutate(
    info = list(image_info(image_read(path))),
    width  = info[[1]]$width,
    height = info[[1]]$height
  ) %>%
  ungroup() %>%
  select(image, width, height)

# ---- 3) Clip bounding boxes to image bounds + drop invalid boxes ----
gt_clip <- gt %>%
  inner_join(img_dims, by = "image") %>%
  mutate(
    # clip to [0, width-1] / [0, height-1]
    TLx_c = pmax(0, pmin(TLx, width  - 1)),
    TLy_c = pmax(0, pmin(TLy, height - 1)),
    BRx_c = pmax(0, pmin(BRx, width  - 1)),
    BRy_c = pmax(0, pmin(BRy, height - 1)),
    
    # enforce ordering (sometimes annotations get swapped)
    TLx_c2 = pmin(TLx_c, BRx_c),
    BRx_c2 = pmax(TLx_c, BRx_c),
    TLy_c2 = pmin(TLy_c, BRy_c),
    BRy_c2 = pmax(TLy_c, BRy_c),
    
    # integer pixel coords (VIAME/Darknet-friendly)
    TLx = as.integer(round(TLx_c2)),
    TLy = as.integer(round(TLy_c2)),
    BRx = as.integer(round(BRx_c2)),
    BRy = as.integer(round(BRy_c2)),
    
    box_w = BRx - TLx,
    box_h = BRy - TLy,
    valid_box = box_w > 0 & box_h > 0
  ) %>%
  filter(valid_box) %>%
  select(image, frame, TLx, TLy, BRx, BRy, label)

message("Rows in original: ", nrow(gt))
message("Rows after clipping + dropping invalid boxes: ", nrow(gt_clip))

# ---- 4) QA/QC: draw boxes on a random image ----
draw_boxes_on_image <- function(img_path, box_df, label_col = NULL) {
  im <- image_read(img_path)
  
  # draw on top of the image
  image_draw(im)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1)) # placeholder; rasterImage uses native dims
  rasterImage(as.raster(im), 0, 0, 1, 1)
  
  # convert image pixel coords to [0,1] plotting coords
  info <- image_info(im)
  W <- info$width
  H <- info$height
  
  # rect() in base graphics uses bottom-left origin, but image coords are top-left.
  # So convert y: y_plot = 1 - (y_pixel / H)
  for (i in seq_len(nrow(box_df))) {
    x1 <- box_df$TLx[i] / W
    x2 <- box_df$BRx[i] / W
    y1 <- 1 - (box_df$TLy[i] / H)
    y2 <- 1 - (box_df$BRy[i] / H)
    
    rect(xleft = x1, ybottom = y2, xright = x2, ytop = y1,
         border = "red", lwd = 2)
    
    if (!is.null(label_col)) {
      text(x = x1, y = y1, labels = box_df[[label_col]][i],
           pos = 4, cex = 0.9, col = "red")
    }
  }
  dev.off()
  
  # Return an annotated magick image object (grab last plot device output)
  # Simpler: re-read from device isn't straightforward; we’ll just save a PNG below.
  invisible(TRUE)
}

# pick a random image that exists
qa_img <- gt_clip %>%
  distinct(image) %>%
  slice_sample(n = 1) %>%
  pull(image)

qa_path <- file.path(IMG_DIR, qa_img)
qa_boxes <- gt_clip %>% filter(image == qa_img)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
qa_out <- file.path(OUT_DIR, paste0("QAQC_", tools::file_path_sans_ext(qa_img), ".png"))

# Save QA plot to file
png(qa_out, width = 1400, height = 900)
draw_boxes_on_image(qa_path, qa_boxes, label_col = "label")
dev.off()

message("Saved QA/QC overlay: ", qa_out)

# ---- 5) Reformat to VIAME-ready groundtruth.csv + labels.txt ----
# VIAME example: id,image,frame,TLx,TLy,BRx,BRy,1,-1,label,1 (headerless)
gt_viame <- gt_clip %>%
  mutate(
    id = row_number(),
    col8 = 1,
    col9 = -1,
    col11 = 1
  ) %>%
  select(id, image, frame, TLx, TLy, BRx, BRy, col8, col9, label, col11)

# Create folder structure VIAME expects (one_per_folder)
dir.create(SEQ_DIR, showWarnings = FALSE, recursive = TRUE)

# Write headerless CSV in seq1/
write.table(
  gt_viame,
  file = file.path(SEQ_DIR, "groundtruth.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

# labels.txt in training root
labels <- sort(unique(gt_viame$label))
writeLines(labels, con = file.path(OUT_DIR, "labels.txt"))

message("Wrote VIAME training files:")
message(" - ", file.path(OUT_DIR, "labels.txt"))
message(" - ", file.path(SEQ_DIR, "groundtruth.csv"))

# OPTIONAL: If you want to copy images into seq1/ automatically:
# file.copy(from = file.path(IMG_DIR, labels of images...), to = SEQ_DIR, overwrite = FALSE)