# Load Data --------------------------------------------------------------------
require(xgboost)
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

# This script demonstrates how to fit generalized linear model in xgboost.
# Basically, we are using linear model, instead of tree for our boosters.
# You can fit a linear regression, or logistic regression model.

# Model Parameters -------------------------------------------------------------
# Change booster to gblinear, so that we are fitting a linear model.
# alpha is the L1 regularizer.
# lambda is the L2 regularizer.
# You can also set lambda_bias which is L2 regularizer on the bias term.
param <- list(
  objective = "binary:logistic", booster = "gblinear",
  nthread = 2, alpha = 0.0001, lambda = 1
)

# Normally, you do not need to set eta (step_size).
# XGBoost uses a parallel coordinate descent algorithm (shotgun).
# There could be affection on convergence with parallelization on certain cases.
# Setting eta to be a smaller value, e.g 0.5 can make the optimization more stable.

# Train Model ------------------------------------------------------------------
# The rest of settings are the same.
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 2
bst <- xgb.train(param, dtrain, num_round, watchlist)

# Prediction -------------------------------------------------------------------
ypred <- predict(bst, dtest)
labels <- getinfo(dtest, "label")
message(paste0("error of preds=", mean(as.numeric(ypred > 0.5) != labels)))
