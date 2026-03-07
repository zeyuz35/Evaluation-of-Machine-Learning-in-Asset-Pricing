library(umap)
library(ggplot2)
library(Rtsne)

final_dataset <- readRDS(here::here("R", "empirical_robustness", "missing_threshold", "final_dataset.rds"))

final_dataset_t <- final_dataset[, names(final_dataset) != "rt", drop = FALSE]

row_sample <- sample(2:18048, 3000, replace = F)
col_sample <- sample(1:740, 500, replace = F)

data.temp <- final_dataset[row_sample, ]

data.temp[is.na(data.temp)] <- 0

data.umap <- umap(data.temp)


d_umap_1 <- as.data.frame(data.umap$layout)

ggplot(d_umap_1, aes(x = V1, y = V2)) +
  geom_point(size = 0.25) +
  guides(colour = guide_legend(override.aes = list(size = 6)))


tsne <- Rtsne(unique(data.temp), dims = 2, perplexity = 35, verbose = TRUE, max_iter = 2000)


## Plotting
d_tsne_1 <- as.data.frame(tsne$Y)
ggplot(d_tsne_1, aes(x = V1, y = V2)) +
  geom_point(size = 0.25) +
  guides(colour = guide_legend(override.aes = list(size = 6)))
