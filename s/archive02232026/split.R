library(imager)
library(stringr)
library(data.table)
# setwd('c:/users/habcam.local/Documents/')
# imagelist1 <- fread('imagelist2224.txt',header=FALSE)
# names(imagelist1) <- 'imagename'
# imagelist1$year <- as.numeric(substr(imagelist1$imagename,8,11))
# imagelist <- subset(imagelist1,year> 2021)
# setwd('d:/')
img_dir <- "../../training2224/split2224/"
imagelist = dat_yr
for (i in 1:length(imagelist$img_name)){
  thisimage <- load.image(paste0(img_dir,imagelist$img_name[i]))
  splitimagelist <- imsplit(thisimage,'x',2)
  ifelse(imagelist$year==2024,side<- 2,side<- 1) #side=1 left, side=2 right
  save.image(splitimagelist[[side]],paste0('../data/raw/',imagelist$img_name[i]))
}
