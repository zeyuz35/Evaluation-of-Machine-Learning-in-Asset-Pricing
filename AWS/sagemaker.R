library("reticulate")
library("aws.s3")
sagemaker <- import('sagemaker')
session <- sagemaker$Session()
role_arn <- session$expand_role('sagemaker-service-role')

# S3
s3_bucket <- session$default_bucket()
s3_prefix <- "simulation_deepar"

## Alow SageMaker to access the S3 bucket

library(tidyverse)

## S3 ##

s3_data_path <- paste0("s3://", bucket, "/data/")
s3_output_path <- paste0("s3://", bucket, "/output/")

## Check if we can read in data
## This works
## Arguments are:
## path = R working directory path, s3_bucket, key_prefix = S3 bucket path
## Note that you can have key_prefix just specify a data folder and it'll download the entire folder

session$download_data("data", s3_bucket, key_prefix = "data/pooled_panel.rds")

pooled_panel <- readRDS("data/pooled_panel.rds")

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
  sagemaker_session = sagemaker_session,
  image_name = image_name,
  role = role,
  train_instance_count = 1,
  train_instance_type = 'ml.c4.2xlarge',
  base_job_name = 'simulation_deepar',
  output_path=s3_output_path
)
  
estimator$set_hyperparameters(num_round = 100L)
job_name <- paste('sagemaker-train-xgboost', format(Sys.time(), '%H-%M-%S'), sep = '-')
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
