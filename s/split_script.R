#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
  library(stringr)
  library(magick)
  library(progress)
  library(tools)
  library(ggplot2)
  library(grid)
})

# ---------------------------------------------------------
# Helper: Parse GEOMETRY_TEXT to Bounding Box
# ---------------------------------------------------------
parse_geometry <- function(geom_text) {
  if (is.na(geom_text) || geom_text == "") {
    return(c(NA, NA, NA, NA))
  }
  
  nums <- as.numeric(unlist(str_extract_all(geom_text, "[-+]?\\d*\\.\\d+|\\d+")))
  
  if (length(nums) >= 4) {
    x1 <- nums[1]; y1 <- nums[2]; x2 <- nums[3]; y2 <- nums[4]
    return(c(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
  }
  return(c(NA, NA, NA, NA))
}

# ---------------------------------------------------------
# Helper: Crop and Save Image
# ---------------------------------------------------------
crop_and_save <- function(src_path, out_path, side) {
  im <- image_read(src_path)
  info <- image_info(im)
  w <- info$width
  h <- info$height
  mid <- w / 2
  
  if (side == "left") {
    geo_string <- sprintf("%dx%d+%d+%d", floor(mid), h, 0, 0)
    offset <- 0
  } else { # right
    geo_string <- sprintf("%dx%d+%d+%d", ceiling(w - mid), h, floor(mid), 0)
    offset <- mid
  }
  
  cropped_im <- image_crop(im, geo_string)
  image_write(cropped_im, path = out_path)
  
  return(list(mid = mid, offset = offset, h = h))
}

# ---------------------------------------------------------
# Helper: Visualize a Subset of Annotations
# ---------------------------------------------------------
visualize_annotations <- function(df, img_dir, out_pdf, n = 5) {
  cat(sprintf("\n🎨 Generating validation visualizations for %d images...\n", n))
  
  # Get unique images that actually have annotations
  valid_images <- unique(df$image)
  if (length(valid_images) == 0) {
    cat("⚠️ No annotations available to visualize.\n")
    return()
  }
  
  sample_imgs <- sample(valid_images, min(n, length(valid_images)))
  
  pdf(out_pdf, width = 10, height = 8)
  
  for (img_name in sample_imgs) {
    img_path <- file.path(img_dir, img_name)
    if (!file.exists(img_path)) next
    
    # Read image for plotting
    im <- image_read(img_path)
    info <- image_info(im)
    w <- info$width
    h <- info$height
    img_grob <- rasterGrob(as.raster(im), interpolate = FALSE)
    
    # Subset annotations for this image
    sub_df <- df %>% filter(image == img_name)
    
    # Plot using ggplot2 with reversed Y-axis (top-left origin)
    p <- ggplot() +
      annotation_custom(img_grob, xmin = 0, xmax = w, ymin = h, ymax = 0) +
      geom_rect(
        data = sub_df,
        aes(xmin = TLx, ymin = TLy, xmax = BRx, ymax = BRy),
        color = "yellow", fill = NA, linewidth = 0.8
      ) +
      coord_fixed(xlim = c(0, w), ylim = c(h, 0), expand = FALSE) +
      labs(
        title = img_name,
        subtitle = sprintf("Validation Plot | %d Annotations | Box Color: Yellow", nrow(sub_df))
      ) +
      theme_void()
    
    print(p)
  }
  
  dev.off()
  cat(sprintf("✅ Visualizations saved to → %s\n", out_pdf))
}

# ---------------------------------------------------------
# Main Processing: Annotations
# ---------------------------------------------------------
process_annotations <- function(csv_path, img_dir, out_img_dir, out_csv, side, visualize, vis_n) {
  dir.create(out_img_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
  
  cat(sprintf("\n📦 Loading annotations from %s...\n", csv_path))
  df <- read_csv(csv_path, show_col_types = FALSE)
  
  df <- df %>%
    rename(image = IMAGE_NAME, label = CLASS_NAME) %>%
    mutate(
      label = tolower(trimws(as.character(label))),
      image = trimws(as.character(image))
    )
  
  parsed <- t(sapply(df$GEOMETRY_TEXT, parse_geometry))
  df$TLx <- parsed[, 1]
  df$TLy <- parsed[, 2]
  df$BRx <- parsed[, 3]
  df$BRy <- parsed[, 4]
  
  df <- df %>% filter(!is.na(TLx) & !is.na(image))
  unique_images <- unique(df$image)
  
  cat(sprintf("\n✂️ Splitting %d ANNOTATED images (%s side)...\n", length(unique_images), toupper(side)))
  pb <- progress_bar$new(total = length(unique_images), format = "[:bar] :percent :eta")
  
  new_rows_list <- list()
  
  for (fname in unique_images) {
    pb$tick()
    src <- file.path(img_dir, fname)
    if (!file.exists(src)) next
    
    out_path <- file.path(out_img_dir, fname)
    
    # Crop the image and get the offset coordinate math
    crop_data <- crop_and_save(src, out_path, side)
    mid <- crop_data$mid
    offset <- crop_data$offset
    
    # 🧠 ADJUST COORDINATES
    sub <- df %>% 
      filter(image == fname) %>%
      mutate(
        # Subtract the offset (0 if left, w/2 if right)
        TLx = TLx - offset,
        BRx = BRx - offset
      ) %>%
      # Ensure boxes are actually on the chosen side
      filter(BRx > 0 & TLx < mid) %>%
      mutate(
        # Clip coordinates so they don't bleed off the edge of the new image
        TLx = pmax(TLx, 0),
        BRx = pmin(BRx, mid),
        bw = BRx - TLx,
        bh = BRy - TLy
      ) %>%
      filter(bw > 1 & bh > 1) %>% 
      mutate(
        image = fname,
        TLx = round(TLx, 2),
        TLy = round(TLy, 2),
        BRx = round(BRx, 2),
        BRy = round(BRy, 2)
      ) %>%
      select(image, TLx, TLy, BRx, BRy, label)
    
    if (nrow(sub) > 0) {
      new_rows_list[[fname]] <- sub
    }
  }
  
  new_df <- bind_rows(new_rows_list)
  write_csv(new_df, out_csv, na = "")
  cat(sprintf("\n💾 Saved %d mapped annotations to → %s\n", nrow(new_df), out_csv))
  
  # TRIGGER VISUALIZATION
  if (visualize && nrow(new_df) > 0) {
    vis_pdf_path <- file.path(dirname(out_csv), sprintf("validation_plot_%s.pdf", side))
    visualize_annotations(new_df, out_img_dir, vis_pdf_path, vis_n)
  }
}

# ---------------------------------------------------------
# Main Processing: Zeros
# ---------------------------------------------------------
process_zeros <- function(csv_path, img_dir, out_img_dir, out_csv, side) {
  dir.create(out_img_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
  
  cat(sprintf("\n📦 Loading zero-annotations from %s...\n", csv_path))
  df <- read_csv(csv_path, show_col_types = FALSE)
  zero_images <- unique(na.omit(df$imagename))
  
  cat(sprintf("\n✂️ Splitting %d ZERO images (%s side)...\n", length(zero_images), toupper(side)))
  pb <- progress_bar$new(total = length(zero_images), format = "[:bar] :percent :eta")
  
  new_rows_list <- list()
  
  for (fname in zero_images) {
    pb$tick()
    src <- file.path(img_dir, fname)
    if (!file.exists(src)) next
    
    out_path <- file.path(out_img_dir, fname)
    crop_and_save(src, out_path, side)
    
    new_rows_list[[fname]] <- data.frame(
      image = fname,
      TLx = NA, TLy = NA, BRx = NA, BRy = NA,
      label = "background",
      stringsAsFactors = FALSE
    )
  }
  
  new_df <- bind_rows(new_rows_list)
  write_csv(new_df, out_csv, na = "")
  cat(sprintf("\n💾 Saved %d background entries to → %s\n", nrow(new_df), out_csv))
}

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
main <- function() {
  option_list <- list(
    make_option("--ann_csv", type = "character"),
    make_option("--ann_img_dir", type = "character"),
    make_option("--out_ann_img_dir", type = "character"),
    make_option("--out_ann_csv", type = "character"),
    
    make_option("--zero_csv", type = "character"),
    make_option("--zero_img_dir", type = "character"),
    make_option("--out_zero_img_dir", type = "character"),
    make_option("--out_zero_csv", type = "character"),
    
    make_option("--side", type = "character", help = "Side to split: 'left' or 'right'"),
    make_option("--visualize", action = "store_true", default = FALSE, help = "Generate a PDF validating the bounding boxes"),
    make_option("--vis_n", type = "integer", default = 5, help = "Number of random images to visualize")
  )
  
  parser <- OptionParser(option_list = option_list)
  opt <- parse_args(parser)
  
  required_args <- c("ann_csv", "ann_img_dir", "out_ann_img_dir", "out_ann_csv", 
                     "zero_csv", "zero_img_dir", "out_zero_img_dir", "out_zero_csv", "side")
  
  if (any(sapply(required_args, function(x) is.null(opt[[x]])))) {
    print_help(parser)
    stop("Missing required arguments.", call. = FALSE)
  }
  
  process_annotations(opt$ann_csv, opt$ann_img_dir, opt$out_ann_img_dir, opt$out_ann_csv, opt$side, opt$visualize, opt$vis_n)
  process_zeros(opt$zero_csv, opt$zero_img_dir, opt$out_zero_img_dir, opt$out_zero_csv, opt$side)
  
  cat("\n✅ All processing complete!\n")
}

main()