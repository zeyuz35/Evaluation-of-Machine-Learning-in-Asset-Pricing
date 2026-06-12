#devtools::install_github("robjhyndman/tsfeatures")
 # required for printing tours
final_dataset <- readRDS("~/Dropbox/website/Evaluation-of-Machine-Learning-in-Asset-Pricing/R/final_dataset.rds")
final_dataset_t<-dplyr::select(final_dataset, -rt)
row_sample<- sample(2:18048,3000, replace=F)
col_sample<- sample(1:740,500, replace=F)
data.temp<- final_dataset[row_sample,]
data.temp[is.na(data.temp)] <- 0
data.umap = umap::umap(data.temp)
d_umap_1 = as.data.frame(data.umap$layout)  
ggplot2::ggplot(d_umap_1, ggplot2::aes(x=V1, y=V2)) +
  ggplot2::geom_point(size=0.25) +
  ggplot2::guides(colour=ggplot2::guide_legend(override.aes=list(size=6)))
tsne <- Rtsne::Rtsne(data.temp, dims = 2, perplexity=35, verbose=TRUE, max_iter = 2000)
## Plotting
d_tsne_1 = as.data.frame(tsne$Y)  
ggplot2::ggplot(d_tsne_1, ggplot2::aes(x=V1, y=V2)) +
  ggplot2::geom_point(size=0.25) +
  ggplot2::guides(colour=ggplot2::guide_legend(override.aes=list(size=6)))
