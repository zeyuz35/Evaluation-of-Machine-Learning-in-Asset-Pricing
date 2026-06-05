require(xgboost)
require(Matrix)
require(data.table)
if (!require(vcd)) {
  install.packages('vcd') #Available in Cran. Used for its dataset with categorical values.
  require(vcd)
}
# According to its documentation, Xgboost works only on numbers.
# Sometimes the dataset we have to work on have categorical data. 
# A categorical variable is one which have a fixed number of values. By example, if for each observation a variable called "Colour" can have only "red", "blue" or "green" as value, it is a categorical variable.
#
# In R, categorical variable is called Factor. 
# Type ?factor in console for more information.
#
# In this demo we will see how to transform a dense dataframe with categorical variables to a sparse matrix before analyzing it in Xgboost.
# The method we are going to see is usually called "one hot encoding".

#load Arthritis dataset in memory.
data(Arthritis)

# create a copy of the dataset with data.table package (data.table is 100% compliant with R dataframe but its syntax is a lot more consistent and its performance are really good).
df <- data.table(Arthritis, keep.rownames = F)

# Let's have a look to the data.table
message("Print the dataset")
print(df)

# 2 columns have factor type, one has ordinal type (ordinal variable is a categorical variable with values wich can be ordered, here: None > Some > Marked).
message("Structure of the dataset")
str(df)

# Let's add some new categorical features to see if it helps. Of course these feature are highly correlated to the Age feature. Usually it's not a good thing in ML, but Tree algorithms (including boosted trees) are able to select the best features, even in case of highly correlated features.

# For the first feature we create groups of age by rounding the real age. Note that we transform it to factor (categorical data) so the algorithm treat them as independant values.
df[,AgeDiscret:= as.factor(round(Age/10,0))]

# Introduce a stronger simplification of the real age with an arbitrary split at 30 years old.
# This illustrates the effect of simplifying information based on arbitrary values.
df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))]

# We remove ID as there is nothing to learn from this feature (it will just add some noise as the dataset is small).
df[,ID:=NULL]

# List the different values for the column Treatment: Placebo, Treated.
message("Values of the categorical feature Treatment")
print(levels(df[,Treatment]))

# Next step, we will transform the categorical data to dummy variables.
# This method is also called one hot encoding.
# The purpose is to transform each value of each categorical feature in one binary feature.
#
# Let's take, the column Treatment will be replaced by two columns, Placebo, and Treated. Each of them will be binary. For example an observation which had the value Placebo in column Treatment before the transformation will have, after the transformation, the value 1 in the new column Placebo and the value 0 in the new column  Treated.
#
# Formulae Improved~.-1 used below means transform all categorical features but column Improved to binary values.
# Column Improved is excluded because it will be our output column, the one we want to predict.
sparse_matrix = sparse.model.matrix(Improved~.-1, data = df)

message("Encoding of the sparse Matrix")
print(sparse_matrix)

# Create the output vector (not sparse)
# 1. Set, for all rows, field in Y column to 0; 
# 2. set Y to 1 when Improved == Marked; 
# 3. Return Y column
output_vector = df[,Y:=0][Improved == "Marked",Y:=1][,Y]

# Following is the same process as other demo
message("Learning...")
bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 9,
               eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic")

importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
print(importance)
# According to the matrix below, the most important feature in this dataset to predict if the treatment will work is the Age. The second most important feature is having received a placebo or not. The sex is third. Then we see our generated features (AgeDiscret). We can see that their contribution is very low (Gain column).

# Evaluate if these results align with statistical significance.
# Calculate the Chi-squared statistic between each feature and the outcome.

print(chisq.test(df$Age, df$Y))
# Pearson correlation between Age and illness disappearing is 35

print(chisq.test(df$AgeDiscret, df$Y))
# Our first simplification of Age gives a Pearson correlation of 8.

print(chisq.test(df$AgeCat, df$Y))
# The arbitrary split between young and old at 30 years old yields a low correlation of 2.
# This result demonstrates that arbitrary assumptions about vulnerability do not align with the data.
# Avoid relying on intuition to construct features; prioritize data-driven approaches.

# In general, destroying information by simplifying it will not improve the model, as demonstrated by the Chi-squared tests.
# In more complex scenarios, creating new features based on existing ones to highlight links with the outcome may help the algorithm.
# However it's almost always worse when you add some arbitrary rules.
# Moreover, you can notice that even if we have added some not useful new features highly correlated with other features, the boosting tree algorithm have been able to choose the best one, which in this case is the Age. Linear model may not be that strong in these scenario.
