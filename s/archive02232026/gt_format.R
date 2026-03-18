# ============================================================
# VIAME groundtruth prep: clip bboxes, QA/QC plot, export
# ============================================================

library(dplyr)
library(readr)
library(stringr)
library(magick)
library(data.table)
library(imager)
library(ggplot2)

# ---- USER EDITS ----
CSV_IN  <- "../data/processed/groundtruth2022_line.csv"   # your current CSV
IMG_DIR <- "../data/images/2022tr_split/"       # folder containing the .png images
OUT_DIR <- "../data/processed/" # output training root
SEQ_DIR <- file.path(OUT_DIR, "seq1")   # VIAME expects a seq folder
set.seed(7)                             # for reproducible random QA/QC pick

gt <- fread(CSV_IN)

# Standardize expected column names if needed
# (edit these if your names differ)
setnames(gt,
         old = c("image","frame","TLx","TLy","BRx","BRy","label"),
         new = c("image","frame","TLx","TLy","BRx","BRy","label"),
         skip_absent = TRUE)

gt[, image := as.character(image)]
gt[, label := as.character(label)]

# unique images table
imgs <- unique(gt[, .(image)])
imgs[, path := file.path(IMG_DIR, image)]
imgs[, exists := file.exists(path)]

if (any(!imgs$exists)) {
  print(imgs[!exists][1:min(.N, 20)])
  warning("Some images referenced in CSV do not exist in IMG_DIR (showing up to 20).")
}

# Function to get (width,height) with minimal retained memory
get_dims <- function(path) {
  tryCatch({
    im <- imager::load.image(path)   # loads one image
    d  <- dim(im)                    # d = (x, y, z, c) where x=width, y=height
    rm(im); gc(FALSE)                # drop it immediately
    list(width = d[1], height = d[2])
  }, error = function(e) {
    list(width = NA_integer_, height = NA_integer_)
  })
}

# Compute dims in a loop (more memory stable than rowwise mutate)
imgs[exists == TRUE, c("width","height") := {
  dims <- lapply(path, get_dims)
  list(
    vapply(dims, `[[`, integer(1), "width"),
    vapply(dims, `[[`, integer(1), "height")
  )
}]

# Inspect failures
bad <- imgs[exists == TRUE & (is.na(width) | is.na(height))]
nrow(bad)

if (nrow(bad) > 0) print(bad[1:min(.N, 20)])

# join dims onto groundtruth
gt2 <- merge(gt, imgs[, .(image, width, height)], by = "image", all.x = TRUE)

# Clip + enforce ordering + drop invalid
gt2[, `:=`(
  TLx = pmax(0, pmin(TLx, width  - 1)),
  TLy = pmax(0, pmin(TLy, height - 1)),
  BRx = pmax(0, pmin(BRx, width  - 1)),
  BRy = pmax(0, pmin(BRy, height - 1))
)]

# enforce TL <= BR
gt2[, `:=`(
  TLx2 = pmin(TLx, BRx),
  BRx2 = pmax(TLx, BRx),
  TLy2 = pmin(TLy, BRy),
  BRy2 = pmax(TLy, BRy)
)]

gt2[, `:=`(
  TLx = as.integer(round(TLx2)),
  TLy = as.integer(round(TLy2)),
  BRx = as.integer(round(BRx2)),
  BRy = as.integer(round(BRy2))
)]

gt2[, `:=`(
  box_w = BRx - TLx,
  box_h = BRy - TLy
)]

gt_clip <- gt2[!is.na(width) & !is.na(height) & box_w > 0 & box_h > 0]

cat("Original rows:", nrow(gt), "\n")
cat("After clipping + filtering:", nrow(gt_clip), "\n")

set.seed(7)

qa_img <- gt_clip[, .N, by = image][sample(.N, 1), image]
qa_path <- file.path(IMG_DIR, qa_img)
qa_boxes <- gt_clip[image == qa_img]

# Load just this one image
im <- imager::load.image(qa_path)
d  <- dim(im)  # width, height

# Convert imager image to data frame for ggplot (can be big; fine for one image)
df_img <- as.data.frame(im)  # columns: x, y, cc, value
# use only one channel if present
df_img <- df_img[df_img$cc == 1, ]

p <- ggplot() +
  geom_raster(data = df_img, aes(x = x, y = y, fill = value)) +
  scale_fill_continuous(guide = "none") +
  coord_equal() +
  # NOTE: imager y increases downward? actually imager y increases downward in plotting;
  # ggplot y increases upward, so we flip y by using (height - y + 1) for the image layer
  scale_y_reverse() +
  geom_rect(
    data = qa_boxes,
    aes(xmin = TLx, ymin = TLy, xmax = BRx, ymax = BRy),
    fill = NA,
    linewidth = 0.7
  ) +
  labs(title = qa_img)
p
OUT_DIR <- "training_data_scallop_2022"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

qa_out <- file.path(OUT_DIR, paste0("QAQC_", qa_img, ".png"))
ggsave(qa_out, p, width = 12, height = 7, dpi = 200)

cat("Saved QA/QC image:", qa_out, "\n")

SEQ_DIR <- file.path(OUT_DIR, "seq1")
dir.create(SEQ_DIR, showWarnings = FALSE, recursive = TRUE)

# VIAME-style: id,image,frame,TLx,TLy,BRx,BRy,1,-1,label,1
gt_viame <- copy(gt_clip)
gt_viame[, id := .I]
gt_viame[, `:=`(col8 = 1, col9 = -1, col11 = 1)]

gt_out <- gt_viame[, .(id, image, frame, TLx, TLy, BRx, BRy, col8, col9, label, col11)]

# Write headerless, no quotes
fwrite(gt_out,
       file = file.path(SEQ_DIR, "groundtruth.csv"),
       sep = ",",
       col.names = FALSE,
       quote = FALSE)

# labels.txt (unique labels)
labs <- sort(unique(gt_out$label))
writeLines(labs, con = file.path(OUT_DIR, "labels.txt"))

cat("Wrote:\n",
    " - ", file.path(OUT_DIR, "labels.txt"), "\n",
    " - ", file.path(SEQ_DIR, "groundtruth.csv"), "\n")
