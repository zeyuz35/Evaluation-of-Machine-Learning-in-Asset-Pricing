######################################################################################################
# INSTRUCTIONS FOR GETTING THIS SETUP WORKING
######################################################################################################
# This is a lot of work, make set up a cloudformation YAML for ease of reproducibility
######################################################################################################
#
# Set up an EC2 instance, doesn't matter too much what's included
# At least t2.medium is recommend due to RAM constraints on the free tier (this is still fairly cheap)
# Note that as most of the actual computation is done via sagemaker, EC2 costs should be quite low
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
  train_instance_count = 1,
  train_instance_type = 'ml.c4.2xlarge',
  base_job_name = 'simulation_deepar',
  output_path = s3_output_path
)
  
estimator$set_hyperparameters(num_round = 100L)
job_name <- paste('sagemaker-simulation-deepar', format(Sys.time(), '%H-%M-%S'), sep = '-')
input_data <- list('train' = s3_train_input,
                   'validation' = s3_valid_input)
estimator$fit(inputs = input_data,
              job_name = job_name)

# S3 Path

estimator$model_data

## Generating Predictions

model_endpoint <- estimator$deploy(initial_instance_count = 1L,
                                   instance_type = 'ml.t2.medium')

model_endpoint$content_type <- 'text/csv'
model_endpoint$serializer <- sagemaker$predictor$csv_serializer

abalone_test <- abalone_test[-1]
num_predict_rows <- 500
test_sample <- as.matrix(abalone_test[1:num_predict_rows, ])
dimnames(test_sample)[[2]] <- NULL

library(stringr)
predictions <- model_endpoint$predict(test_sample)
predictions <- str_split(predictions, pattern = ',', simplify = TRUE)
predictions <- as.numeric(predictions)

abalone_test <- cbind(predicted_rings = predictions, 
                      abalone_test[1:num_predict_rows, ])
head(abalone_test)  

## Delete endpoint to minimize endpoint costs

session$delete_endpoint(model_endpoint$endpoint)
