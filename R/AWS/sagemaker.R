######################################################################################################
# INSTRUCTIONS FOR GETTING THIS SETUP WORKING
######################################################################################################
# This is a lot of work, make set up a cloudformation YAML for ease of reproducibility
######################################################################################################
#
# Set up an EC2 instance, doesn't matter too much what's included
# At least t2.medium is recommend due to RAM constraints on the free tier (this is still fairly cheap)
# Note that as most of the actual computation is done via sagemaker, EC2 costs should be quite low
# Sagemaker fees are quite expensive though unfortunately
# I preferred an Ubuntu image because there's a bit more documentation
# 
# Create the following IAM roles:
# 
# For Sagemaker:
# sagemaker-service-role
# AmazonSageMakerFullAccess
# AmazonEC2ContainerRegistryFullAccess
# 
# For EC2:
# ec2-rstudio-sagemaker
# IAMReadOnlyAccess
# AmazonSageMakerFullAccess
# 
# Create the following security group:
# rstudio-sagemaker-sg
# Open up ports:
# 22, 80, 8787
# 
# Attach the EC2 IAM role and security group to your EC2 instance
# 
# SSH into your EC2 instance
# Install the latest version of R using these commands
# UBUNTU
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
# sudo apt update
# sudo apt install r-base
# 
# Install the latest version of Rstudio Server
# UBUNTU
# sudo apt-get install gdebi-core
# wget https://download2.rstudio.org/server/bionic/amd64/rstudio-server-1.2.5033-amd64.deb
# sudo gdebi rstudio-server-1.2.5033-amd64.deb
# 
# Install sagemaker, boto3, etc if necessary
# sudo pip install sagemaker boto3
# 
# Add an extra rstudio user so that you can log into rstudio server, and follow the prompts
# sudo adduser rstudio
# 
# Log into Rstudio Server, go to the terminal tab
# Run the following to get the region name set correctly
# mkdir /home/rstudio/.aws
# region=`curl http://169.254.169.254/latest/dynamic/instance-identity/document|grep region|awk -F\" '{print $4}'`
# echo "[default]" > /home/rstudio/.aws/config
# echo "region =" $region >> /home/rstudio/.aws/config
# 
# Restart Rstudio just to refresh it (doesn't seem to be necessary though)
# 
########################################################################################################
# ACTUAL CODE START
########################################################################################################

library("reticulate")
## Force reticulate to use the right version of anaconda and thus sagemaker modules
use_condaenv("/home/ubuntu/anaconda3", required = TRUE)
sagemaker <- import('sagemaker')
boto3 <- import("boto3")

boto3$Session()$region_name

session <- sagemaker$Session()

role_arn <- session$expand_role('sagemaker-service-role')

# S3
# This creates an S3 bucket with a default name
# You could also specify your own S3 bucket, but this may be a bit easier
s3_bucket <- session$default_bucket()
s3_prefix <- "simulation_deepar"

## Alow SageMaker to access the S3 bucket

library(tidyverse)

## S3 ##

s3_data_path <- paste0("s3://", s3_bucket, "/data/")
s3_output_path <- paste0("s3://", s3_bucket, "/output/")

## Check if we can read in data
## This works
## Arguments are:
## path = R working directory path, s3_bucket, key_prefix = S3 bucket path
## Note that you can have key_prefix just specify a data folder and it'll download the entire folder

session$download_data("data", s3_bucket, key_prefix = "data/pooled_panel.rds")

session$download_data("data", s3_bucket, key_prefix = "data/pooled_panel.json")

pooled_panel <- readRDS("data/pooled_panel.rds")

###
## Convert this to a JSON format which sagemaker can understand
###

## Sagemaker requires a JSON format, where each time series is its own "line", 
## and each line consists of the time index, as well as the value (target), 
## and dynamic features (our regressors in this case)

## E.g. 
## {"start": "2016-01-01 00:00:00", "target": [1, 2, 3], "dynamic_feat": [[x11, x12, x13], [x21, x22, x23], ...]}
## {"start": "2016-01-01 00:00:00", "target": [1, 2, 3]}... etc

## Note that the examples given in the notebooks only have one dynamic feature, but deepar can support more

## DeepAR can also support missing data in the target series (y variable), but not the dyanmic features
## This makes it a little strange to work with for our dataset

###########################################################################
library(jsonlite)
###########################################################################
## Function that takes data frame in tidy format, and outputs a dataframe 
## that is ready to be exported to JSON for use with sagemaker deepar

tidy_to_json <- function(data, start) {
  stock_id <- data$stock %>%
    unique()
  
  # Number of cross sectional units
  cross_units <- length(stock_id)
  
  ## JUst setting the beginning time to something arbitrary for now, change if needed
  pooled_panel_json <- data.frame(start = rep(start, cross_units), 
                                  target = c(1:cross_units), 
                                  dynamic_feat = c(1:cross_units))
  
  for (i in 1:cross_units) {
    pooled_panel_filter <- data %>%
      filter(stock == stock_id[i])
    
    pooled_panel_json$target[i] <- list(pooled_panel_filter$rt)
    
    pooled_panel_filter_feature <- pooled_panel_filter %>%
      select(-time, -rt, -stock) %>%
      unname() %>%
      as.matrix() %>%
      # Transpose it to get the right format of one feature series per row
      t()
    
    pooled_panel_json[i, ]$dynamic_feat <- pooled_panel_filter_feature %>% list()
  }
  pooled_panel_json
}

pooled_panel_train <- pooled_panel %>%
  filter(time <= 168) %>%
  tidy_to_json(start = "2000-01-01 00:00:00")

pooled_panel_test <- pooled_panel %>%
  filter(time > 168) %>%
  tidy_to_json(start = "2014-01-01 00:00:00")

pooled_panel_train %>%
  stream_out(file("./data/pooled_panel_train.json"))

pooled_panel_test %>%
  stream_out(file("./data/pooled_panel_test.json"))

## Upload to S3, note that this function uploads the entire directory you specify

session$upload_data("data", s3_bucket, key_prefix = "data")

#######################################
## Get the image required for deepar ##
#######################################

region <- session$boto_region_name

image_name = sagemaker$amazon$amazon_estimator$get_image_uri(region, "forecasting-deepar", "latest")

#####################
## Train a Model
#####################

# Define an estimator

estimator <- sagemaker$estimator$Estimator(
  sagemaker_session = session,
  image_name = image_name,
  role = role_arn,
  train_instance_count = 1L,
  train_instance_type = 'ml.c4.xlarge',
  base_job_name = 'simulation_deepar',
  output_path = s3_output_path
)

# Lags are used in the model building procedure anyway, so context_length doesn't have to be very large
# Amazon recommends to just set this equal to prediction length
context_length <- 12L
prediction_length <- 12L
freq <- "1M"
  
estimator$set_hyperparameters(
  time_freq = "1M",
  context_length = context_length,
  prediction_length = prediction_length,
  num_cells = 40L,
  num_layers = 4L,
  likelihood = "student-T",
  epochs = 80L,
  mini_batch_size = 128L,
  learning_rate = "0.0001",
  dropout_rate = 0.15,
  early_stopping_patience = 20L,
  num_dynamic_feat = "auto",
  test_quantiles = array(0.5)
)

s3_train_input <- "s3://sagemaker-us-west-2-438078873022/data/pooled_panel_train.json"
s3_valid_input <- "s3://sagemaker-us-west-2-438078873022/data/pooled_panel_test.json"

data_channels <- list('train' = s3_train_input, 'test' = s3_valid_input)

job_name <- paste('sagemaker-simulation-deepar', format(Sys.time(), '%H-%M-%S'), sep = '-')
estimator$fit(inputs = data_channels,
              job_name = job_name)

############################################################################
## HYPERPARAMETER TUNING
############################################################################

## NOT RUN, as it seems fairly robust to hyperparameters (ie they don't help much), 
## and very costly and time consuming

## Import some functions from

# HyperparameterTuner <- sagemaker$tuner$HyperparameterTuner
# IntegerParameter <- sagemaker$tuner$IntegerParameter
# CategoricalParameter <- sagemaker$tuner$CategoricalParameter
# ContinuousParameter <- sagemaker$tuner$ContinuousParameter
# 
# hyperparameter_ranges <- list(
#   learning_rate = ContinuousParameter(0.0001, 1),
#   num_cells = IntegerParameter(10L, 100L),
#   num_layers = IntegerParameter(1L, 4L),
#   dropout_rate = ContinuousParameter(0, 0.2)
# )
# 
# objective_metric_name <- "test:mean_wQuantileLoss"
# 
# tuner = HyperparameterTuner(estimator,
#                             objective_metric_name,
#                             objective_type = 'Minimize',
#                             hyperparameter_ranges,
#                             max_jobs = 96L,
#                             max_parallel_jobs = 3L)
# 
# tuner$fit(list(train = s3_train_input, test = s3_valid_input), include_cls_metadata = FALSE)

##############################################################################################################
##############################################################################################################
#### Generating Predictions
##############################################################################################################
##############################################################################################################

###########################
## MODEL ENDPOINT APPROACH
##############################
## An endpoint is like a server which you can send requests to and receive predictions in real time
## Obviously, designed with real use business cases in mind
## Downside - quite unintuitive for "normal" usage, and may not work with large number of time series
## Implemented anyway, as this is the recommended approach

################
## DATA PREP
################
## Function that takes a tidy dataframe, and produce a dataframe in the right format
## to be converted to the inference JSON format to make predictions
## Require the tidy_to_json function from before
## Remember that this returns the DATAFRAME, and to get the actual text format you need to pass to through toJSON

## Note that DeepAR takes entire time series and produces predictions 
## according to the prediction length argument you specified when building the model
## Therefore, provide it with the training + validation set data (assuming this isn't too computationally intensive)

json_to_inference_json <- function(json_df) {
  inf_json <- list(
    instances = list(start = pooled_panel_json$start, 
                     target = pooled_panel_json$target, 
                     dynamic_feat = pooled_panel_json$dynamic_feat),
    configuration = list(
      num_samples = 10, output_types = c("mean", "quantiles", "samples"), quantiles = c("0.5")
    )
  )
}

tidy_to_inf_json <- function(dataframe, start) {
  dataframe_json <- dataframe %>% tidy_to_json(start = start)
  
  dataframe_json %>% json_to_inference_json()
}

####################################
## Set up model endpoint object
####################################

model_endpoint <- estimator$deploy(initial_instance_count = 1L,
                                   instance_type = 'ml.t2.medium',
                                   content_type="application/json")

## Wrapper function that takes a tidy_df and 

deepar_predict <- function(tidy_df) {
  model_endpoint$predict(tidy_df %>% tidy_to_inf_json(start = "2000-01-01 00:00:00"))
}

##################################################
## Delete the endpoint afterwards to save $$$

session$delete_endpoint(model_endpoint$endpoint)

######################################################
## BATCH TRANSFORM APPROACH
######################################################
## Much more analagous to "normal" functions, this "transforms" input data to output data using a trained sagemaker model
## Also seems to be much more straightforward and well suited to large number of time series

## Specify inference input and output path

batch_input <- paste0("s3://", s3_bucket, "/batch/input/pooled_panel_inference_in.json")
batch_output <- paste0("s3://", s3_bucket, "/batch/output/pooled_panel_inference_out.json")

transformer = model.transformer(instance_count=1, instance_type='ml.m4.xlarge', output_path = batch_output)

transformer.transform(data=batch_input, data_type='S3Prefix', content_type='text/csv', split_type='Line')

transformer.wait()


