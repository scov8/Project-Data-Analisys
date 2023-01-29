### Ragionando su quali regressori utilizzare ###

# Scartiamo a priori la tecnica di OLS poichè l'ordine di grandezza di p è circa pari a quello di n,
# quindi la varianza potrebbe aumentare e raggiungere l'overfitting.
# In questi casi, può essere più opportuno utilizzare metodi di regressione più robusti o regolarizzati,
# come la regressione ridge o la regressione lasso, che possono aiutare a gestire il problema di overfitting
# e migliorare la robustezza del modello.
# Prendiamo i valori dei coefficienti dei predictors che hanno un t-value maggiore e un Pr[|t|] minore;


# dato che l'ordine di grandezza di p è circa uguale a quella di n applicare OLS non è 
# l'ideale perché la varianza diventa più grande e possiamo avere overfitting oltre a 
# produrre stime completamente sballata

# Quando abbiamo un gran numero di predittori X nel modello, generalmente ce ne saranno molti 
# che hanno scarso effetto o addirittura nessun effetto su Y. Lasciare queste variabili nel 
# modello rende più difficile vedere le relazioni reali tra i predittori e la variabile dipendente 
# (non vediamo bene il "quadro generale"), ed è difficile apprezzare l'effetto delle 
# "variabili rilevanti" che descrivono Y. Il modello sarebbe più facile da interpretare rimuovendo 
# le variabili non importanti, ovvero impostando i loro coefficienti a zero


####################################################
####################### Setup ######################
####################################################

library(glmnet)

# Carichiamo il dataset
data = read.csv("RegressionDataset_DA_group1.csv", header=T, na.strings ="?")

# Creo i due set
set.seed(1985) # Utilizziamo un seed fissato per la riprodurre l'esperimento
sample <- sample.int(n = nrow(data), size = floor(.8*nrow(data)), replace = F) # Settiamo una percentuale dell'80%
training_set <- data[sample,] # Nel training set mettiamo il primo 80% dei valori
test_set <- data[-sample,] # Nel test set mettiamo l'ultimo 20% dei valori

# Salvo in x e y i valori dell'intero dataset
x = model.matrix(Y~., data)[,-1] # Salvo in x tutti i valori tranne quelli di Y
y = data$Y # Salvo in y i valori della colonna Y

# salvo in x_train e y_train i valori del training set
x_train = model.matrix(Y~., training_set)[,-1]
y_train = training_set$Y

# salvo in x_test e y_test i valori del test set
x_test = model.matrix(Y~., test_set)[,-1]
y_test = test_set$Y


# a. confrontare tra loro le tecniche per la costruzione di modelli empirici lineari presentate al corso,
# scartando quelle che non `e opportuno utilizzare per questo tipo di data set;

####################################################
###################### Ridge #######################
####################################################
# By default the glmnet() function automatically selects range of lambda values. A lambda set overrides default, a decreasing sequence of lambda values is provided from (10^10 to 10^-2 <- close to 0)
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
# By default glmnet() standardizes the variables so that they are on the same scale. To turn off this default setting, use the argument standardize=FALSE.
# Associated with each value of lambda is a vector of ridge regression coefficients, stored in a matrix that can be accessed by coef(), in this case 20x100, 19+intercept for each lambda value:
dim(coef(ridge.mod))
# We expect the coefficient estimates to be much smaller, in terms of l2 norm, when a large value of lambda is used, as compared to when a small value is used.
ridge.mod$lambda[50] # grid[50] = 11497.57
coef(ridge.mod)[,50] # corresponding coefficients
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # l2 norm
ridge.mod$lambda[60] # lambda = 705.48
coef(ridge.mod)[,60] # corresponding coefficients
sqrt(sum(coef(ridge.mod)[-1,60]^2)) # l2 norm > l2 for lambda[50]
# obtain the ridge regression coefficients for a new lambda, say 50:
# ?predict.glmnet #for help
predict(ridge.mod,s=50,type="coefficients")[1:51,]

####################################################
#### Validation approach to estimate test error ####
####################################################
# fit a ridge regression model on the training set, and evaluate its MSE on the test set, using lambda = 4.
ridge.mod=glmnet(x_train,y_train,alpha=0,lambda=grid,thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x_test) # Note the use of the predict() function again. This time we get predictions for a test set, by replacing type="coefficients" with the newx argument.
mean((ridge.pred-y_test)^2) # test MSE
mean((mean(y_train)-y_test)^2) # test MSE, if we had instead simply fit a model with just an intercept, we would have predicted each test observation using the mean of the training observations.
# We could also get the same result by fitting a ridge regression model
# with a very large value of lambda, i.e. 10^10:
ridge.pred=predict(ridge.mod,s=1e10,newx=x_test)
mean((ridge.pred-y_test)^2) # like intercept only
# Least squares is simply ridge regression with lambda=0;
ridge.pred=predict(ridge.mod,s=0,newx=x_test,exact=T,x=x_train,y=y_train) # corrected according to errata (glmnet pack updated)
# In order for glmnet() to yield the exact least squares coefficients when lambda = 0, we use the argument exact=T when calling the predict() function. Otherwise, the predict() function will interpolate over the grid of lambda values used in fitting the glmnet() model, yielding approximate results. When we use exact=T, there remains a slight discrepancy in the third decimal place between the output of glmnet() when lambda = 0 and the output of lm(); this is due to numerical approximation on the part of glmnet().
mean((ridge.pred-y_test)^2)
# Compare the results from glmnet when lambda=0 with lm()
lm(y_train~x_train)
predict(ridge.mod,s=0,exact=T,type="coefficients",x=x_train,y=y_train)[1:51,] # corrected according to errata
# In general, if we want to fit a (unpenalized) least squares model, then we should use the lm() function, since that function provides more useful outputs,such as standard errors and p-values.
model <- lm(y_train~x_train)
summary(model)
coeff_ridge<-coef(model)

####################################################
################### Print Results ##################
####################################################

# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save
top_n <- which(coeff_ridge >= sort(coeff_ridge, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_ridge <- round(coeff_ridge[top_n]/100, digits = 0)
intToUtf8(predictors_ridge)

####################################################
################ CROSS-VALIDATION ##################
####################################################
# Instead of using the arbitrary value lambda=4, cv.glmnet() uses cross-validation to choose the tuning parameter
# By default, the function performs ten-fold cross-validation,
# though this can be changed using the argument nfolds.
set.seed (1985)
cv.out=cv.glmnet(x_train,y_train,alpha=0)
dev.new()
plot(cv.out)
bestlam=cv.out$lambda.min; bestlam; log(bestlam) # the best lambda (212 on the text)
cv.out$lambda.1se # one standard error rule
log(cv.out$lambda.1se)
ridge.pred = predict(ridge.mod,s=bestlam ,newx=x_test)
mse_ridge <- mean((ridge.pred-y_test)^2); mse_ridge
# This represents a further improvement over the test MSE when lambda=4.
# Finally refit our ridge regression model on the full data set with the best lambda
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:51,]
# As expected, none of the coefficients are zero
# ridge regression does not perform variable selection!
dev.new()
plot(out,label = T, xvar = "lambda")
#install.packages("plotmo")
library(plotmo)
dev.new()
plot_glmnet(out)

####################################################
###################### LASSO #######################
####################################################
# use the argument alpha = 1 to perform lasso
lasso.mod = glmnet(x_train, y_train, alpha=1, lambda=grid)
dev.new()
plot(lasso.mod,label = T)
dev.new()
plot(lasso.mod,label = T, xvar = "lambda")
dev.new(); plot_glmnet(lasso.mod, xvar = "lambda")
# perform cross-validation
set.seed (1985)
cv.out=cv.glmnet(x_train,y_train,alpha=1)
dev.new()
plot(cv.out)
bestlam=cv.out$lambda.min; print(bestlam);print(log(bestlam))
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))
lasso.pred=predict(lasso.mod,s=bestlam ,newx=x_test)
mse_lasso <- mean((lasso.pred-y_test)^2); mse_lasso

# wrt lm
lasso.pred=predict(lasso.mod,s=0,newx=x_test,exact=T,x=x_train,y=y_train)
mse_lm <- mean((lasso.pred-y_test)^2); mse_lm
# However, the lasso has a substantial advantage:
# some of the 19 coefficient estimates are exactly zero (12 on the text).
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:51,]
lasso.coef
lasso.coef[lasso.coef!=0]
cat("Number of coefficients equal to 0:",sum(lasso.coef==0),"\n")

lasso.coef=predict(out,type="coefficients",s=0)[1:51,] # lm as lasso for lambda=0
lasso.coef
coeff_lasso <- coef(lm(Y~., data=data)) # differences on the 3rd place


###### lm full model #######
mse_full <- mean((predict(lm(Y~., data=training_set),newdata = test_set)-y_test)^2)
summary(lm(Y~., data=test_set))


####################################################
################### Print Results ##################
####################################################

# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save, based on the previous parameters
top_n <- which(coeff_lasso >= sort(coeff_lasso, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_lasso <- round(coeff_lasso[top_n]/100, digits = 0)
intToUtf8(predictors_lasso)


####################################################
################### Elastic Net ####################
####################################################
#for (alp in seq(0.1,0.9,by=0.1)) {
alp <- 0.95
enet.mod = glmnet(x_train, y_train, alpha=alp, lambda=grid)
dev.new()
plot(enet.mod,label = T)
dev.new()
plot(enet.mod,label = T, xvar = "lambda")
plot_glmnet(enet.mod, xvar = "lambda")
# CV
set.seed (1985)
cv.out=cv.glmnet(x_train, y_train,alpha=alp,nfolds = 5)
dev.new()
plot(cv.out)
bestlam_enet=cv.out$lambda.min; print(bestlam_enet);print(log(bestlam_enet))
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))
enet.pred=predict(enet.mod,s=bestlam_enet ,newx=x_test)
mse_enet <- mean((enet.pred-y_test)^2)
print(mse_enet)
#}
predict(enet.mod,type="coefficients",s=bestlam_enet)[1:51,]
coeff_elasticNet <- coef(lm(Y~., data=data)) # differences on the 3rd place


####################################################
################### Print Results ##################
####################################################

# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save, based on the previous parameters
top_n <- which(coeff_elasticNet >= sort(coeff_elasticNet, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_elasticNet <- round(coeff_elasticNet[top_n]/100, digits = 0)
intToUtf8(predictors_elasticNet)
