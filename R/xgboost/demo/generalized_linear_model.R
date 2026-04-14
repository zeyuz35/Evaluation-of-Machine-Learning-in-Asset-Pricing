# Section Setup ----------------------------------------------------------------
require(xgboost)

# Load in the agaricus dataset.
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

# Section GLM Demonstration ----------------------------------------------------
# This script demonstrates how to fit a generalized linear model in xgboost.
# We are using a linear model instead of a tree for our boosters.
# You can fit a linear regression or a logistic regression model.

# Change booster to gblinear so that we are fitting a linear model.
# The alpha parameter is the L1 regularizer.
# The lambda parameter is the L2 regularizer.
# You can also set lambda_bias which is the L2 regularizer on the bias term.
param <- list(
  objective = "binary:logistic", booster = "gblinear",
  nthread = 2, alpha = 0.0001, lambda = 1
)

# Normally you do not need to set eta (step size).
# XGBoost uses a parallel coordinate descent algorithm (shotgun).
# There could be affection on convergence with parallelization in certain cases.
# Setting eta to a smaller value (e.g. 0.5) can make optimization more stable.

# The rest of the settings are the same.
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 2
bst <- xgb.train(param, dtrain, num_round, watchlist)
ypred <- predict(bst, dtest)
labels <- getinfo(dtest, "label")
message(paste0("error of preds=", mean(as.numeric(ypred > 0.5) != labels)))
