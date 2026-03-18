library(stringr)

img_names <- unique(anno$Imagename)

dates_chr <- str_extract(img_names, "(?<=\\.)\\d{8}(?=\\.)")

dates <- as.Date(dates_chr, format = "%Y%m%d")

range(dates, na.rm = TRUE)
summary(dates)
table(dates)

plot(table(dates), type = "h", lwd = 2,
     main = "Images per day",
     xlab = "Date", ylab = "Count")

df_dates <- data.frame(
  image = img_names,
  date = dates
)


library(dplyr)

df_dates |> 
  filter(date < as.Date("2024-05-15") & date > as.Date("2024-05-01")) |>
  head(n = 10)
