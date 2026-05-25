#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
  library(stringr)
  library(magick)
  library(ggplot2)
  library(grid)
})

# ---------------------------------------------------------
# Helper: Parse GEOMETRY_TEXT to Line & BBox
# ---------------------------------------------------------
parse_geometry_detailed <- function(geom_text) {
  if (is.na(geom_text) || geom_text == "") {
    return(c(x1=NA, y1=NA, x2=NA, y2=NA, TLx=NA, TLy=NA, BRx=NA, BRy=NA))
  }
  
  nums <- as.numeric(unlist(str_extract_all(geom_text, "[-+]?\\d*\\.\\d+|\\d+")))
  
  if (length(nums) >= 4) {
    x1 <- nums[1]; y1 <- nums[2]; x2 <- nums[3]; y2 <- nums[4]
    return(c(
      x1 = x1, y1 = y1, x2 = x2, y2 = y2,
      TLx = min(x1, x2), TLy = min(y1, y2), 
      BRx = max(x1, x2), BRy = max(y1, y2)
    ))
  }
  return(c(x1=NA, y1=NA, x2=NA, y2=NA, TLx=NA, TLy=NA, BRx=NA, BRy=NA))
}

# ---------------------------------------------------------
# Helper: Load Path Mapping from Manifest
# ---------------------------------------------------------
load_path_mapping <- function(txt_file) {
  cat(sprintf("📄 Loading source paths from %s...\n", txt_file))
  paths <- readLines(txt_file, warn = FALSE)
  paths <- paths[paths != ""] # Remove empty lines
  setNames(paths, basename(paths))
}

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
main <- function() {
  option_list <- list(
    make_option("--ann_csv", type = "character", help = "Path to the raw annotations CSV"),
    make_option("--src_txt", type = "character", help = "Path to the source images manifest TXT"),
    make_option("--out_pdf", type = "character", help = "Output PDF path (e.g., validation_plot.pdf)"),
    make_option("--n_images", type = "integer", default = 20, help = "Number of random images to visualize")
  )
  
  parser <- OptionParser(option_list = option_list)
  opt <- parse_args(parser)
  
  if (is.null(opt$ann_csv) || is.null(opt$src_txt) || is.null(opt$out_pdf)) {
    print_help(parser)
    stop("Missing required arguments (--ann_csv, --src_txt, --out_pdf).", call. = FALSE)
  }
  
  path_mapping <- load_path_mapping(opt$src_txt)
  
  cat(sprintf("📦 Loading annotations from %s...\n", opt$ann_csv))
  df <- read_csv(opt$ann_csv, show_col_types = FALSE)
  
  # Normalize columns
  if ("IMAGE_NAME" %in% colnames(df)) {
    df <- df %>% rename(image = IMAGE_NAME, label = CLASS_NAME)
  }
  
  # Parse geometry for both lines and boxes
  parsed <- t(sapply(df$GEOMETRY_TEXT, parse_geometry_detailed))
  df <- cbind(df, as.data.frame(parsed))
  
  df <- df %>% filter(!is.na(TLx) & !is.na(image))
  
  # Filter only images that exist in our manifest and on disk
  available_images <- unique(df$image)
  available_images <- available_images[available_images %in% names(path_mapping)]
  available_images <- available_images[file.exists(path_mapping[available_images])]
  
  if (length(available_images) == 0) {
    stop("No images from the CSV were found using the provided manifest.", call. = FALSE)
  }
  
  # Sample images
  n_to_sample <- min(opt$n_images, length(available_images))
  sample_imgs <- sample(available_images, n_to_sample)
  
  cat(sprintf("🎨 Generating validation PDF for %d images...\n", n_to_sample))
  
  dir.create(dirname(opt$out_pdf), recursive = TRUE, showWarnings = FALSE)
  pdf(opt$out_pdf, width = 12, height = 8)
  
  for (img_name in sample_imgs) {
    src <- path_mapping[img_name]
    
    # Read image
    im <- image_read(src)
    info <- image_info(im)
    w <- info$width
    h <- info$height
    img_grob <- rasterGrob(as.raster(im), interpolate = FALSE)
    
    # Subset annotations
    sub_df <- df %>% filter(image == img_name)
    
    p <- ggplot() +
      annotation_custom(img_grob, xmin = 0, xmax = w, ymin = h, ymax = 0) +
      # Plot the converted bounding box
      geom_rect(
        data = sub_df,
        aes(xmin = TLx, ymin = TLy, xmax = BRx, ymax = BRy),
        color = "yellow", fill = NA, linewidth = 0.8
      ) +
      # Plot the original annotated line segment
      geom_segment(
        data = sub_df,
        aes(x = x1, y = y1, xend = x2, yend = y2),
        color = "cyan", linewidth = 1.2, alpha = 0.8
      ) +
      coord_fixed(xlim = c(0, w), ylim = c(h, 0), expand = FALSE) +
      labs(
        title = img_name,
        subtitle = sprintf("Original Stereo Image | %d Annotations\nCyan = Original Line | Yellow = Derived Bounding Box", nrow(sub_df))
      ) +
      theme_void()
    
    print(p)
  }
  
  dev.off()
  cat(sprintf("✅ Visualizations saved to → %s\n", opt$out_pdf))
}

main()