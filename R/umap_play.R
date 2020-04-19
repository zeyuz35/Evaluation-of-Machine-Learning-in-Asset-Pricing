library(umap)
library(purrr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(Rtsne)
#devtools::install_github("robjhyndman/tsfeatures")
library(tsfeatures)
library(gganimate) # required for printing tours
library(sneezy)
final_dataset <- readRDS("~/Dropbox/website/Evaluation-of-Machine-Learning-in-Asset-Pricing/R/final_dataset.rds")

final_dataset_t<-final_dataset%>%select(-rt)

row_sample<- sample(2:18048,3000, replace=F)
col_sample<- sample(1:740,500, replace=F)

data.temp<- final_dataset[row_sample,]

data.temp[is.na(data.temp)] <- 0

data.umap = umap(data.temp)


d_umap_1 = as.data.frame(data.umap$layout)  

ggplot(d_umap_1, aes(x=V1, y=V2)) +  
  geom_point(size=0.25) +
  guides(colour=guide_legend(override.aes=list(size=6)))


tsne <- Rtsne(data.temp, dims = 2, perplexity=35, verbose=TRUE, max_iter = 2000)


## Plotting
d_tsne_1 = as.data.frame(tsne$Y)  
ggplot(d_tsne_1, aes(x=V1, y=V2)) +  
  geom_point(size=0.25) +
  guides(colour=guide_legend(override.aes=list(size=6)))











