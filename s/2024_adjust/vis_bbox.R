library(imager)
library(data.table)

# -------------------------
# Paths
# -------------------------
img_dir <- "/Volumes/PortableSSD/saltnoaa/images/2024tr/"
anno_csv <- "../data/raw/groundtruth24.csv"

# -------------------------
# List images
# -------------------------
img_files <- list.files(
  img_dir,
  pattern = "\\.(png|jpg|jpeg|tif|tiff)$",
  full.names = FALSE,
  ignore.case = TRUE
)

# normalize
img_files <- tolower(trimws(img_files))

# -------------------------
# Load annotations
# -------------------------
anno <- fread(anno_csv)

# normalize
anno[, Imagename := tolower(trimws(Imagename))]
anno[, ClassName := tolower(trimws(ClassName))]

# filter to scallops only (optional)
anno <- anno[ClassName == "scallop"]

# keep only images that exist
anno <- anno[Imagename %in% img_files]

cat("Unique images after filtering:", length(unique(anno$Imagename)), "\n")

# -------------------------
# Sample images
# -------------------------
set.seed(4)
sample_imgs <- sample(unique(anno$Imagename), 20)

# -------------------------
# Plot function
# -------------------------
plot_image_with_boxes <- function(img_name) {
  
  img_path <- file.path(img_dir, img_name)
  
  if (!file.exists(img_path)) {
    message("Missing: ", img_name)
    return(NULL)
  }
  
  im <- load.image(img_path)
  
  sub <- anno[Imagename == img_name]
  
  plot(im, main = img_name)
  
  for (i in 1:nrow(sub)) {
    rect(
      xleft   = sub$TLx[i],
      ybottom = sub$TLy[i],
      xright  = sub$BRx[i],
      ytop    = sub$BRy[i],
      border  = "red",
      lwd     = 2
    )
  }
}

# -------------------------
# Display 20 images
# -------------------------
# par(mfrow = c(4,5), mar = c(1,1,2,1))

for (img in sample_imgs) {
  plot_image_with_boxes(img)
}


sample_imgs
imgfile = sample_imgs[3]
imgfile = "202403.20240514.225635676.10616.png"
jpeg(file = paste("~/Downloads/",imgfile,sep=""), width = 1600, height = 1200)
plot_image_with_boxes(imgfile)
dev.off()
ggplot2::ggsave(source_anno, 
                file = "~/Downloads/202404.20240627.234633098.20390.jpg", device = "jpg")


################################################################


shift_boxes_x <- function(df, shift_px = 50, img_width = NULL) {
  
  df[, `:=`(
    TLx_shift = TLx + shift_px,
    BRx_shift = BRx + shift_px
  )]
  
  if (!is.null(img_width)) {
    df[, `:=`(
      TLx_shift = pmax(0, TLx_shift),
      BRx_shift = pmin(img_width, BRx_shift)
    )]
  }
  
  return(df)
}

# anno_shifted <- shift_boxes_x(anno, shift_px = 75)

plot_image_with_shift <- function(img_name, shift_px = 50) {
  
  img_path <- file.path(img_dir, img_name)
  im <- load.image(img_path)
  
  sub <- anno[Imagename == img_name]
  sub_shift <- shift_boxes_x(copy(sub), shift_px)
  
  plot(im, main = paste0(img_name, " | shift=", shift_px))
  
  # original boxes (red)
  for (i in 1:nrow(sub)) {
    rect(
      sub$TLx[i], sub$TLy[i],
      sub$BRx[i], sub$BRy[i],
      border = "grey", lwd = 1
    )
  }
  
  # shifted boxes (blue)
  for (i in 1:nrow(sub_shift)) {
    
    if (any(is.na(c(
      sub_shift$TLx_shift[i],
      sub_shift$TLy[i],
      sub_shift$BRx_shift[i],
      sub_shift$BRy[i]
    )))) next
    
    rect(
      sub_shift$TLx_shift[i], sub_shift$TLy[i],
      sub_shift$BRx_shift[i], sub_shift$BRy[i],
      border = "blue", lwd = 2
    )
  }
}

set.seed(1)
sample_imgs <- sample(unique(anno$Imagename), 10)
sample_imgs
# par(mfrow = c(2,5), mar = c(1,1,2,1))

# for (img in sample_imgs) {
#   plot_image_with_shift(img, shift_px = 75)
# }

imgfile = sample_imgs[10]
jpeg(file = paste("~/Downloads/",imgfile,sep=""), width = 1600, height = 1200)
plot_image_with_shift(imgfile, shift_px = 55)
dev.off()
