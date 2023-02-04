####################################################
####################### Setup ######################
####################################################
library(glmnet)   # Load the "glmnet" package
library(plotmo)   # Load the "plotmo" package
library(corrplot) # Load the "corrplot" package
library(psych)    # Load the "psych" package
library(leaps)    # Load the "leaps" package for BSS and stepwise
library(car)      # Load the "car" package for Variance Inflation Factor (VIF) calculation

# Load the dataset
data <- read.csv("RegressionDataset_DA_group1.csv", header = T, na.strings = "?")

head(data)
data <- na.omit(data) # Remove rows with missing values

set.seed(2023)        # Use a fixed seed for reproducibility of the experiment

# Save "Y" values in "y" and all values except for "Y" in "x"
x <- model.matrix(Y ~ ., data)[, -1] # Save all values except for "Y"
y <- data$Y # Save "Y" values

train <- sample(1:nrow(x), 0.8 * nrow(x)) # Sample 80% of the data for training set
test <- (-train)  # The remaining data will be the test set
y.test <- y[test] # Save test set "Y" values

# Parameters
n <- nrow(data[train, ]) # Number of observations in the training set
p <- ncol(x[train, ])    # Number of regressors

####################################################
############### Preliminary analysis ###############
####################################################

############### Correlation Matrix #################
dev.new()
corData <- round(cor(x), digits = 2) # Calculate the correlation matrix of x and round the values to 2 decimal places
# Uncomment the next line to save the plot as a PDF file
# pdf(file="corr.pdf", width=26, height=15)
corPlot(corData, cex = 0.22, show.legend = TRUE, main = "Correlation Matrix") # Plot the correlation matrix with specified font size and display legend
# dev.off()

####################################################
################## Linear Model ####################
####################################################
lm.mod <- lm(Y ~ ., data = data[train, ]) # Fit a linear regression model to the data, using all variables in training set
summary(lm.mod) # Get the summary of the model
lm.predict <- predict(lm.mod, newdata = (data[test, ])) # Make predictions using the model on the test data
lm.coef <- coef(lm.mod) # Get the coefficients of the model

####################### Test ######################
lm.mse <- mean((lm.predict - y[test])^2) # Calculate the mean squared error between the predicted values and the actual values in the test data
lm.mse # Display the mean squared error

################### Print Results ##################
lm.predictors <- round(lm.coef / 100, digits = 0) # Round the values in bwd.coef and divide by 100, with 0 decimal places
intToUtf8(lm.predictors) # Convert the values in bwd.predictors to UTF-8 encoded integers

# vif(lm.mod) # verifica la collinearità dei dati #CANCELLARE?????????

####################################################
###################### BSS #########################
####################################################
# The following code performs BSS, which is computationally inefficient, however, since it both has a p
# approximately greater than 30 
# full.regfit <- regsubsets(Y ~ ., nvmax = variables_selected, data = data[train, ], really.big = T)
# summary(full.regfit)

####################################################
################### Stepwise #######################
####################################################
# The following code performs forward, backward, and hybrid selection on the training data 
# using the 'regsubsets' function from the 'leaps' package. 
# The function returns the best subset of predictors for each method of selection.
# The performance of each method is evaluated by computing the mean squared error (MSE) 
# between the predicted values and the actual values of the response variable on the test data.
# Finally, the coefficients of the best models are rounded to the nearest 100 and converted to UTF-8 format.

################# Forward selection ################
fwd.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "forward") # Forward selection on the training data
summary(fwd.regfit) # Summary of the results of forward selection

################# Backward selection ################
bwd.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "backward") # Backward selection on the training data
summary(bwd.regfit) # Summary of the results of forward selection

################# Hybrid selection ################
hyb.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "seqrep") # Hybrid selection on the training data
summary(hyb.regfit) # Summary of the results of forward selection

####################### Test ######################
test.mat <- model.matrix(Y ~ ., data = data[test, ]) #Model matrix for the test data

############ Test on backward selection ############
# The following lines evaluate the performance of the best model obtained from backward selection
val.errors <- rep(NA, p) # Creates a vector of NA values with length 'p' to store the MSE values for each model

for (i in 1:p) {
  coefi <- coef(bwd.regfit, id = i)           # Extracts the coefficients for the i-th model obtained from backward selection
  pred <- test.mat[, names(coefi)] %*% coefi  # Computes the predicted values for the response variable
  val.errors[i] <- mean((y[test] - pred)^2)   # Computes the MSE for the i-th model and stores it in the 'val.errors' vector
}

min <- which.min(val.errors) # Index of the minimum MSE value
bwd.mse <- val.errors[min]   # Stores the minimum MSE value in 'bwd.mse'
bwd.coef <- coef(bwd.regfit, min) # Extracts the coefficients for the best model obtained from backward selection

########### Test on forward selection ###########
# The following lines evaluate the performance of the best model obtained from forward selection
val.errors <- rep(NA, p) # Creates a vector of NA values with length 'p' to store the MSE values for each model

for (i in 1:p) {
  coefi <- coef(fwd.regfit, id = i)           # Extracts the coefficients for the i-th model obtained from forward selection
  pred <- test.mat[, names(coefi)] %*% coefi  # Computes the predicted values for the response variable
  val.errors[i] <- mean((y[test] - pred)^2)   # Computes the MSE for the i-th model and stores it in the 'val.errors' vector
}

min <- which.min(val.errors) # Index of the minimum MSE value
fwd.mse <- val.errors[min]   # Stores the minimum MSE value in 'fwd.mse'
fwd.coef <- coef(fwd.regfit, which.min(val.errors)) # Extracts the coefficients for the best model obtained from forward selection

############ Test on hybrid selection ############
# The following lines evaluate the performance of the best model obtained from hybrid selection
val.errors <- rep(NA, p) # Creates a vector of NA values with length 'p' to store the MSE values for each model

for (i in 1:p) {
  coefi <- coef(hyb.regfit, id = i)           # Extracts the coefficients for the i-th model obtained from hybrid selection
  pred <- test.mat[, names(coefi)] %*% coefi  # Computes the predicted values for the response variable
  val.errors[i] <- mean((y[test] - pred)^2)   # Computes the MSE for the i-th model and stores it in the 'val.errors' vector
}

# The best model is the one that contains which.min(val.errors) (ten in the book) variables.
min <- which.min(val.errors) # Index of the minimum MSE value
hyb.mse <- val.errors[min] # Stores the minimum MSE value in 'hyb.mse'
hyb.coef <- coef(hyb.regfit, which.min(val.errors)) # Extracts the coefficients for the best model obtained from hybrid selection

######################## Plot ########################

###### LI LASCIAMO???????????????????????!!!!
dev.new()
# pdf(file="backward_cp_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "Cp")
title("Backward selection with Cp")
# dev.off()

dev.new()
# pdf(file="forward_cp_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "Cp")
title("Forward selection with Cp")
# dev.off()

dev.new()
# pdf(file="hybrid_cp_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "Cp")
title("Hybrid selection with Cp")
# dev.off()

dev.new()
# pdf(file="backward_bic_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "bic")
title("Backward selection with bic")
# dev.off()

dev.new()
# pdf(file="forward_bic_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "bic")
title("Forward selection with bic")
# dev.off()

dev.new()
# pdf(file="hybrid_bic_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "bic")
title("Hybrid selection with bic")
# dev.off()

dev.new()
# pdf(file="backward_adjr_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "adjr2")
title("Backward selection with adjr2")
# dev.off()

dev.new()
# pdf(file="forward_adjr_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "adjr2")
title("Forward selection with adjr2")
# dev.off()

dev.new()
# pdf(file="hybrid_adjr_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "adjr2")
title("Hybrid selection with adjr2")
# dev.off()

################### Print Results ##################
bwd.predictors <- round(bwd.coef / 100, digits = 0) # Round the values in bwd.coef and divide by 100, with 0 decimal places
intToUtf8(bwd.predictors) # Convert the values in bwd.predictors to UTF-8 encoded integers

fwd.predictors <- round(fwd.coef / 100, digits = 0) # Round the values in fwd.coef and divide by 100, with 0 decimal places
intToUtf8(fwd.predictors) # Convert the values in fwd.predictors to UTF-8 encoded integers

hyb.predictors <- round(hyb.coef / 100, digits = 0) # Round the values in hyb.coef and divide by 100, with 0 decimal places
intToUtf8(hyb.predictors) # Convert the values in hyb.predictors to UTF-8 encoded integers

####################################################
###################### Ridge #######################
####################################################
grid <- 10^seq(5, -2, length = 1000) # Define a sequence of 1000 numbers logarithmically spaced between 10^5 and 10^-2

# Use the argument alpha = 0 to perform ridge
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid) # Fit a ridge regression model with alpha=0 and the grid of lambda values defined earlier

dev.new()
plot_glmnet(ridge.mod, xvar = "lambda") # Plot the ridge regression model, with x-axis as log lambda

# Perform 10-fold cross-validation to determine the best lambda value for the ridge model
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0, lambda = grid)

dev.new()
plot(cv.out) # Plot the cross-validation results

bestlam <- cv.out$lambda.min # Get the best lambda value from cross-validation

ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ]) # Calculate the predictions using the best lambda value
ridge.mse <- mean((ridge.pred - y[test])^2) # Calculate the mean squared error of the predictions on test set
ridge.mse # Print the mean squared error

ridge.out <- glmnet(x, y, alpha = 0, lambda = grid) # Fit the ridge regression model to the entire data set
ridge.coef <- predict(ridge.out, type = "coefficients", s = bestlam)[1:p + 1, ] # Get the coefficients for the best lambda value
ridge.coef # Print the coefficients

dev.new()
plot_glmnet(ridge.out, xvar = "lambda") # Plot the ridge regression model for the entire data set

################### Print Results ##################
# Round the coefficients and convert to utf-8 format
ridge.predictors <- round(ridge.coef / 100, digits = 0)
intToUtf8(ridge.predictors)

####################################################
###################### LASSO #######################
####################################################
grid <- 10^seq(5, -2, length = 1000) # Define a sequence of 1000 numbers logarithmically spaced between 10^5 and 10^-2

# use the argument alpha = 1 to perform lasso
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid) # Fit a lasso regression model with alpha=1 and the grid of lambda values defined earlier

dev.new()
plot(lasso.mod, label = T) # Plot the model fit to the training data as a function of L1 norm

dev.new()
plot_glmnet(lasso.mod, xvar = "lambda") # Plot the model fit to the training data as a function of log lambda using plot_glmnet() function

# Perform 10-fold cross-validation to determine the best lambda value for the lasso model
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1, lambda = grid)

dev.new()
plot(cv.out) # Plot the cross-validation results

bestlam <- cv.out$lambda.1se # Determine the best lambda value based on the cross-validation results, we choose the 1se to select
                             # the most parsimonious model with low error 
# bestlam <- cv.out$lambda.min # If you want the minimum only based on MSE

lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ]) # Use the best lambda value to make predictions on the test data
lasso.mse <- mean((lasso.pred - y[test])^2) # Calculate the mean squared error of the predictions on the test data
lasso.mse # Print the mean squared error

lasso.out <- glmnet(x, y, alpha = 1, lambda = grid) # Fit the lasso model to the entire data set
lasso.coef <- predict(lasso.out, type = "coefficients", s = bestlam)[1:p + 1, ] # Extract coefficients
lasso.coef # Print the coefficient values
lasso.coef[lasso.coef != 0] # Print the non-zero coefficient values
cat("Number of coefficients equal to 0:", sum(lasso.coef == 0), "\n") # Print the number of coefficients equal to 0

dev.new()
plot_glmnet(lasso.out, xvar = "lambda") # Plot the lasso model fit to the entire data set as a function of log lambda using plot_glmnet() function

################### Print Results ##################
# Round the coefficient values to the nearest 100 and convert the result to UTF-8
lasso.predictors <- round(lasso.coef / 100, digits = 0)
intToUtf8(lasso.predictors)

####################################################
################### Elastic Net ####################
####################################################
enet.mod <- glmnet(x[train, ], y[train], alpha = 0.1, lambda = grid) # Fit the Elastic Net Model with an alpha value of 0.1
enet.pred <- predict(enet.mod, s = 0.1, newx = x[test, ]) # Predict using the fitted model and a lambda value of 0.1
min <- mean((enet.pred - y.test)^2) # Calculate the mean squared error
def.alp <- 0 # Initialize default alpha value

# Loop through different alpha values from 0.5 to 0.9 with a step of 0.05
# This loop is more towards the lasso side since it's not recommended to apply ridge for highly uncorrelated data
for (alp in seq(0.5, 0.9, by = 0.05)) {
  enet.mod <- glmnet(x[train, ], y[train], alpha = alp, lambda = grid) # Fit the Elastic Net Model with the current alpha value
  cv.out <- cv.glmnet(x[train, ], y[train], alpha = alp, lambda = grid) # Perform cross-validation to find the best lambda value
  enet.bestlam <- cv.out$lambda.min # Get the best lambda value
  
  enet.pred <- predict(enet.mod, s = enet.bestlam, newx = x[test, ]) # Predict using the fitted model and the best lambda value
  enet.mse <- mean((enet.pred - y.test)^2) # Calculate the mean squared error
  
  # If the current mean squared error is lower than the minimum mean squared error, update the minimum and default alpha value
  if (enet.mse < min) {
    min <- enet.mse
    def.alp <- alp
  }
}

# Fit the Elastic Net Model with the best default alpha value
enet.mod <- glmnet(x[train, ], y[train], alpha = def.alp, lambda = grid)
# Plot the fitted model
dev.new()
plot(enet.mod, label = T)
dev.new()
plot(enet.mod, label = T, xvar = "lambda")
plot_glmnet(enet.mod, xvar = "lambda")

cv.out <- cv.glmnet(x[train, ], y[train], alpha = def.alp, lambda = grid) # Perform cross-validation

# Plot the cross-validation results
dev.new()
plot(cv.out)

enet.bestlam <- cv.out$lambda.min # Get the best lambda value

enet.pred <- predict(enet.mod, s = enet.bestlam, newx = x[test, ]) # Predict using the fitted model and the best lambda value
enet.mse <- mean((enet.pred - y.test)^2) # Calculate the mean squared error

enet.coef <- predict(enet.mod, type = "coefficients", s = enet.bestlam)[1:p+1, ] # Get the coefficients for the fitted model with the best lambda value

################### Print Results ##################
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
enet.predictors <- round(enet.coef / 100, digits = 0)
intToUtf8(enet.predictors)

####################################################
################# Print all MSE ###################
####################################################
# Finally, we print all calculated MSE values to evaluate the best technique
lm.mse
fwd.mse
bwd.mse
hyb.mse
lasso.mse
ridge.mse
enet.mse
