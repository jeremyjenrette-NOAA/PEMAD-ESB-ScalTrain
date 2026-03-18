library(imager)
library(data.table)
library(stringr)

# -------------------------
# Paths
# -------------------------
img_dir <- "/Volumes/PortableSSD/saltnoaa/images/2024tr/"
anno_csv <- "../data/raw/annotations_2024.csv"

img_files <- list.files(
  img_dir,
  pattern = "\\.(png|jpg|jpeg|tif|tiff)$",
  full.names = FALSE,
  ignore.case = TRUE
)
# -------------------------
# Load annotations
# -------------------------
anno1 <- fread(anno_csv)

# keep only line annotations
anno1 <- anno1[geom_type == "line"]

# extract image filename
anno1[, imagename := trimws(imagename)]
img_files <- trimws(img_files)

# optional but recommended
anno1[, imagename := tolower(imagename)]
img_files <- tolower(img_files)
anno1 <- anno1[imagename %in% img_files]
cat("Unique annotated images (line only):", length(unique(anno1$imagename)), "\n")
cat("Images on disk:", length(img_files), "\n")
# -------------------------
# Parse geometry_text
# -------------------------
parse_line <- function(txt) {
  # extract numbers
  nums <- as.numeric(unlist(str_extract_all(txt, "[0-9\\.]+")))
  # should be: x1 y1 x2 y2
  return(nums[1:4])
}

coords <- t(sapply(anno1$geometry_text, parse_line))
colnames(coords) <- c("x1", "y1", "x2", "y2")

anno1 <- cbind(anno1, coords)

# -------------------------
# Sample 20 random images
# -------------------------
set.seed(9)
sample_imgs <- sample(unique(anno1$imagename), 20)

# -------------------------
# Plot function
# -------------------------
plot_image_with_lines <- function(img_name) {
  
  img_path <- file.path(img_dir, img_name)
  
  if (!file.exists(img_path)) {
    message("Missing: ", img_name)
    return(NULL)
  }
  
  im <- load.image(img_path)
  
  # subset annotations for this image
  sub <- anno1[imagename == img_name]
  
  # plot
  plot(im, main = img_name)
  
  for (i in 1:nrow(sub)) {
    segments(
      x0 = sub$x1[i],
      y0 = sub$y1[i],
      x1 = sub$x2[i],
      y1 = sub$y2[i],
      col = "red",
      lwd = 2
    )
  }
}

# -------------------------
# Display 20 images
# -------------------------
# par(mfrow = c(1,5), mar = c(1,1,2,1))

for (img in sample_imgs) {
  plot_image_with_lines(img)
}

imgfile = "202403.20240505.030020131.69244.png"
jpeg(file = paste("~/Downloads/",imgfile,sep=""), width = 800, height = 600)
plot_image_with_lines(imgfile)
dev.off()
ggplot2::ggsave(source_anno, 
                file = "~/Downloads/202404.20240627.234633098.20390.jpg", device = "jpg")
