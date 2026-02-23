library(tools)

#-------------------------------#
# Inputs
#-------------------------------#
# IMG_DIR_FULL <- "/Volumes/PortableSSD/saltnoaa/images/2022tr/"
IMG_DIR_FULL   <- "../data/images/2022tr_split/"
dat_dir = "../data/raw/"
anndat <- read.csv("../data/raw/parsedann2224.csv", stringsAsFactors = FALSE)
#-------------------------------#
# 1) Filter annotations to 2022 scallops (bbox-ready rows)
#-------------------------------#
anndat_scal_2022 <- subset(
  anndat,
  year == 2022 & Spname == "Scallop" & anntype == "line"
)

# Ensure filenames match what's on disk (strip any paths if present)
anndat_scal_2022$IMAGE_NAME <- basename(anndat_scal_2022$IMAGE_NAME)

#-------------------------------#
# 2) List images actually present in the 2022 directory
#    (adjust extensions if needed)
#-------------------------------#
img_files <- list.files(
  IMG_DIR_FULL,
  pattern = "\\.(png|jpg|jpeg|tif|tiff)$",
  full.names = FALSE,
  ignore.case = TRUE
)

# Keep only annotations whose image exists in the directory
anndat_scal_2022 <- anndat_scal_2022[anndat_scal_2022$IMAGE_NAME %in% img_files, ]

# Optional sanity checks
cat("Images on disk:", length(unique(img_files)), "\n")
cat("Annotated images matched:", length(unique(anndat_scal_2022$IMAGE_NAME)), "\n")
cat("Total scallop boxes (2022):", nrow(anndat_scal_2022), "\n")

#-------------------------------#
# 3) Build groundtruth rows
#    Format (tab-delimited, no header), like your example:
#    [0] row_id
#    [1] IMAGE_NAME
#    [2] frame (0)
#    [3] TLx
#    [4] TLy
#    [5] BRx
#    [6] BRy
#    [7] (1)
#    [8] (-1)
#    [9] label ("scallop")
#    [10] (1)
#-------------------------------#

# Coerce bbox coords to integers (and in case they come in as character)
to_int <- function(x) as.integer(round(as.numeric(x)))
anndat_scal_2022 = ann %>% filter(geom_type == "line" & imagename %in% img_files)
gt <- data.frame(
  row_id = seq_len(nrow(anndat_scal_2022)) - 1L,
  image  = anndat_scal_2022$IMAGE_NAME,
  frame  = 0L,
  TLx    = to_int(anndat_scal_2022$TLx),
  TLy    = to_int(anndat_scal_2022$TLy),
  BRx    = to_int(anndat_scal_2022$BRx),
  BRy    = to_int(anndat_scal_2022$BRy),
  col8   = 1L,
  col9   = -1L,
  label  = "scallop",
  col11  = 1L,
  stringsAsFactors = FALSE
)

# Optional: drop any rows with missing/invalid coords
gt <- gt[complete.cases(gt[, c("TLx","TLy","BRx","BRy")]), ]
gt <- gt[gt$BRx > gt$TLx & gt$BRy > gt$TLy, ]

#-------------------------------#
# 4) Write groundtruth.csv (tab-delimited, no header)
#-------------------------------#
out_file <- file.path(dat_dir, "groundtruth.csv")
write.table(
  gt,
  file = out_file,
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

cat("Wrote:", out_file, "\n")
