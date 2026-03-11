setwd("./VIAME_Habcam/PEMAD-ESB-ScalTrain/s")
dat = read.csv("../data/raw/gt2224_corrected.csv")
colnames(dat)
head(dat)
######################################################################
library(dplyr)
library(stringr)

dat <- dat %>%
  mutate(
    year = str_sub(img_name, 1, 4) |> as.integer()
  )

table(dat$year)

img_dir <- "../../training2224/split2224/"
yr_pick = 2022

dat_yr <- subset(dat, year == yr_pick)

sc <- dat_yr %>%
  filter(spname == "Scallop") %>%
  mutate(
    img_path = file.path(img_dir, img_name),
    box_w = BRx - TLx,
    box_h = BRy - TLy,
    box_area = box_w * box_h
  )

# Core summaries
summary_core <- list(
  n_boxes_total   = nrow(sc),
  n_images        = n_distinct(sc$img_path),
  mean_boxes_img  = nrow(sc) / n_distinct(sc$img_name),
  missing_images   = sum(!file.exists(unique(sc$img_path)))
)

summary_core

# per image QA sampling
sc_by_img <- sc %>%
  group_by(img_name) %>%
  summarise(
    n_boxes = n(),
    area_med = median(box_area, na.rm = TRUE),
    area_mean = mean(box_area, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_boxes))

head(sc_by_img, 20)     # most “crowded” scallop images
summary(sc_by_img$n_boxes)

################################################################################
# distribution plots
library(ggplot2)

ggplot(sc_by_img, aes(n_boxes)) +
  geom_histogram() +
  labs(title = "Scallop boxes per image", x = "Boxes per image", y = "Image count")

ggplot(sc, aes(box_area)) +
  geom_histogram() +
  scale_x_log10() +
  labs(title = "Scallop box area (log10 scale)", x = "Box area", y = "Count")

ggplot(sc, aes(box_w)) +
  geom_histogram() +
  scale_x_log10() +
  labs(title = "Scallop box width (log10)", x = "Box width (pixels)", y = "Count")
################################################################################
library(magick)
library(dplyr)
library(purrr)
library(stringr)

img_dir <- "../../training2224/split2224/"
out_dir <- "../figures/montages"   # change as needed
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

TARGET_YEAR <- yr_pick
SEED <- 23
N_PICK <- 6

# Filter scallops, pick images
sc_y <- sc %>% filter(year == TARGET_YEAR)

sc_by_img_y <- sc_y %>%
  group_by(img_name) %>%
  summarise(n_boxes = n(), .groups = "drop")

set.seed(SEED)
pick_imgs <- sc_by_img_y %>%
  filter(file.exists(file.path(img_dir, img_name))) %>%
  slice_sample(n = N_PICK) %>%
  pull(img_name)

draw_boxes_one <- function(img_name, df, img_dir, color = "red") {
  this <- df %>% filter(img_name == .env$img_name)
  path <- file.path(img_dir, img_name)
  
  im <- image_read(path)
  
  for (i in seq_len(nrow(this))) {
    im <- image_draw(im)
    rect(
      xleft   = this$TLx[i],
      ytop    = this$TLy[i],
      xright  = this$BRx[i],
      ybottom = this$BRy[i],
      border  = color,
      lwd     = 3
    )
    dev.off()
  }
  
  im
}

ims <- map(pick_imgs, draw_boxes_one, df = sc_y, img_dir = img_dir)
pick_imgs
montage <- image_montage(
  image_join(ims),
  tile = "3x2",
  geometry = "640x"
)
montage
# Save it
outfile <- file.path(out_dir, sprintf("montage_Scallop_%d_seed%d_n%d.png", TARGET_YEAR, SEED, N_PICK))
image_write(montage, path = outfile, format = "png")

outfile

