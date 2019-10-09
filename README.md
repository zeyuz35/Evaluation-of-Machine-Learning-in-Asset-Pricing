# Evaluation-of-Machine-Learning-in-Asset-Pricing

GitHub Repository for my honours thesis at Monash University

Topic: Evaluation of Machine Learning in Asset Pricing

Supervisor: David Frazier

This thesis aims to evaluate the application of machine learning methods to the problems of prediction and causal inference in empirical finance, with specific regards to low signal to noise ratios and regressors which exhibit persistence and cross sectional correlation. This is explored via a simulation study and an empirica study.

All code is in the R folder, and all files related to the actual thesis are in the Thesis folder.

# Simulation

Simulation code should be finished, in "Simulation.rmd". This includes flexible options to specify different parameters and properties. Generation of each panel dataset in a panel dataframe format, along with diagnostics such as various r squared values are all possible with provided functions. This section is to be run to generate the actual datasets first.

"Simulation Models.rmd" contains all the functions required to fit models to the datasets. Coded up to be as modular as possible, so in theory you could just specify a single dataset and run each model individually in this section (indeed this is what I've done to test them).

"Simulation Model Results.rmd" contains wrapper functions that run any combination models, calling them from "Simulation Models.rmd" These will spit out a list object containing all specified model results. This also includes all functions to extract the results to produce csv files containing all results, and plotting them in various ways.

Generally speaking, the project has been coded up in an efficient manner. Simulation, model fitting, variable importance calculation etc are all parallelized in addition to using efficient packages for maximum effiency. A notable exception are the variable importance functions for the neural networks, which does not work with foreach due to difficulties with reticulate.

Note that the random forests and neural networks are VERY computationally intensive. On an i5 6600K @ 4.4GHz + NVidia GTX 960 running a set of either of these typically requires a least a few hours, often overnight.

# Real Dataset

The majority of the real dataset is provided by Dacheng Xiu, available from his website http://dachxiu.chicagobooth.edu/ under "Empirical Asset Pricing via Machine Learning." This dataset notably excludes the RET (holding period return) or Price columns, which have to be queried from WRDS/CRSP separately. Code to combine these two elements is provided.

This was then combined with Welch Goyal's dataset containing various macroeconomic factors, available from Amit Goyal's website http://www.hec.unil.ch/agoyal/. Specifically, the updated dataset up until 2018 was used. 
