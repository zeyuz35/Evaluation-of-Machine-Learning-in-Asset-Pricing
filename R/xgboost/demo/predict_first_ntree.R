# Section Setup ----------------------------------------------------------------
require(xgboost)

# Load in the agaricus dataset
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

# Section Training -------------------------------------------------------------
param <- list(max_depth = 2, eta = 1, silent = 1, objective = "binary:logistic")
watchlist <- list(eval = dtest, train = dtrain)
nrounds <- 2

# Training the model for two rounds
bst <- xgb.train(param, dtrain, nrounds, nthread = 2, watchlist)

# Section Prediction -----------------------------------------------------------
message("start testing prediction from first n trees")
labels <- getinfo(dtest, "label")

# Predict using first 1 tree
ypred1 <- predict(bst, dtest, ntreelimit = 1)

# By default, we predict using all the trees
ypred2 <- predict(bst, dtest)

message("error of ypred1 = ", mean(as.numeric(ypred1 > 0.5) != labels))
message("error of ypred2 = ", mean(as.numeric(ypred2 > 0.5) != labels))
