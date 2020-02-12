# Evaluation-of-Machine-Learning-in-Asset-Pricing

GitHub Repository for my honours thesis at Monash University

Topic: Evaluation of Machine Learning in Asset Pricing

Supervisor: David Frazier

This thesis aims to evaluate the application of machine learning methods to the problems of prediction and causal inference in empirical finance, with specific regards to low signal to noise ratios and regressors which exhibit persistence and cross sectional correlation. This is explored via a simulation study and an empirica study.

All code is in the R folder, and all files related to the actual thesis are in the Thesis folder.

# Simulation

Simulation code should be finished, in "Simulation.rmd". This includes flexible options to specify different parameters and properties. Generation of each panel dataset in a panel dataframe format, along with diagnostics such as various r squared values are all possible with provided functions. This section is to be run to generate the actual datasets first.

"Simulation Models.rmd" contains all the functions required to fit models to the datasets. Coded up to be as modular as possible, so in theory you could just specify a single dataset and run each model individually in this section (indeed this is what I've done to test them).

"Simulation Models_run.rmd" contains all the wrapper functions to iteratively loop over all the simulation realisations, and will spit out list objects that contain all specified model results. Model results are already provided in .rds format in the repo.

"Simulation Model_results.rmd" contains all the code to load in the results and produce pretty tables (in LaTeX, or optionally html with a bit of fiddling) and plots. Some functions are also available to extract/average over results in certain ways.

Generally speaking, the project has been coded up in an efficient manner. Simulation, model fitting, variable importance calculation etc are all parallelized in addition to using efficient packages for maximum effiency. There are some exceptions to this where the implementation of the methods does not play nicely with parallelization (e.g. python packages run through "reticulate" on Windows, or random forest variable importance functions which are already parallelized).

Note that the random forests and neural networks are VERY computationally intensive. On an i5 6600K @ 4.4GHz + NVidia GTX 960 running a set of either of these typically requires a least a few hours, often overnight.

# Real Dataset

The majority of the real dataset is provided by Dacheng Xiu, available from his website http://dachxiu.chicagobooth.edu/ under "Empirical Asset Pricing via Machine Learning." This dataset notably excludes the RET (holding period return) or Price columns, which have to be queried from WRDS/CRSP separately. Code to combine these two elements is provided.

This was then combined with Welch Goyal's dataset containing various macroeconomic factors, available from Amit Goyal's website http://www.hec.unil.ch/agoyal/. Specifically, the updated dataset up until 2018 was used. 

For robustness checks the Fama French Five Factors were also used, available from Kenneth French's website https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.

## FFORMA

An alternative implementation of FFORMA (https://github.com/robjhyndman/M4metalearning and https://www-sciencedirect-com.ezproxy.lib.monash.edu.au/science/article/pii/S0169207019300895) was produced for ease of use with data in a tidy format. Because FFORMA requires running XGBoost with a custom loss function which is not officially supported, instructions to clone and install a fixed fork of XGBoost are provided.

## Amazon DeepAR

This part of the project needs to be run on an AWS EC2 instance with permissions to an S3 bucket. There are some alternative implementations of this, but they are unofficial and very tricky to compile and get running.

Instructions with how to set up the required AWS services up and running are included in the AWS folder. This includes commented instructions for how to set up an Ubuntu Machine Learning Server hosting Rstudio Server, as well as many wrapper functions to get the data generated from R into a JSON format which Sagemaker likes.

Note that AWS is quite costly - running this part of the project cost ~160 USD (mostly from EC2 and Sagemaker instances). 
