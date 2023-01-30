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
library(plotmo)

# Carichiamo il dataset
data = read.csv("RegressionDataset_DA_group1.csv", header=T, na.strings ="?")

head(data)
# unique(unlist (lapply (Auto, function (x) which (is.na (x))))) #To find all the rows in a data frame with at least one NA
data=na.omit(data)

# Creo i due set
set.seed(1985) # Utilizziamo un seed fissato per la riprodurre l'esperimento

# Salvo in x e y i valori dell'intero dataset
x = model.matrix(Y~., data)[,-1] # Salvo in x tutti i valori tranne quelli di Y
y = data$Y # Salvo in y i valori della colonna Y

train=sample(1:nrow(x), 0.8*nrow(x)) # another typical approach to sample
test=(-train)
y.test=y[test]

#parameters
grid=10^seq(10,-2,length=100)
lambda_values <- seq(from = 0.0001, to = 10, by = 0.001)
set.seed (1985)

####################################################
###################### Ridge #######################
####################################################
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid) # si esegue la refressione ridge
dev.new()
plot(ridge.mod,label = T)
dev.new()
plot(ridge.mod,label = T, xvar = "lambda")
dev.new(); plot_glmnet(ridge.mod, xvar = "lambda")
dim(coef(ridge.mod)) # dimensione dei coefficienti

############# perform cross-validation #############
# in particolare si usa un approccio di k-fold cross validation
cv.out=cv.glmnet(x[train,],y[train],alpha=0, nfolds=10)
dev.new()
plot(cv.out)
bestlam=cv.out$lambda.min; print(bestlam);print(log(bestlam))
# 1se sta per one standard error
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))


# da eliminare
ridge.pred = predict(ridge.mod, s=10, newx=x[test,])
min <- mean((ridge.pred-y.test)^2); mse_ridge
lambda_def=10
i=0
i
lambda_def
min
for (i in 1:length(lambda_values)) {
  # Effettuiamo la predizione utilizzando Ridge col migliore lambda
  ridge.pred = predict(ridge.mod, s=lambda_values[i], newx=x[test,])
  mse_ridge <- mean((ridge.pred-y.test)^2); mse_ridge
  if(mse_ridge<=min){
    min=mse_ridge
    lambda_def=lambda_values[i]
  }
}
i
lambda_def
min
# end eliminre

# Effettuiamo la predizione utilizzando Ridge con lambda=0
ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
mse_lm <- mean((ridge.pred-y.test)^2); mse_lm

# This represents a further improvement over the test MSE when lambda=4.
# Finally refit our ridge regression model on the full data set with the best lambda
out=glmnet(x,y,alpha=0)
ridge.coef = predict(out,type="coefficients",s=0)[1:51,]

# As expected, none of the coefficients are zero
# ridge regression does not perform variable selection!
dev.new()
plot(out,label = T, xvar = "lambda")
#install.packages("plotmo")

dev.new()
plot_glmnet(out)

####################################################
################### Print Results ##################
####################################################

# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save
top_n <- which(ridge.coef >= sort(ridge.coef, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_ridge <- round(ridge.coef[top_n]/100, digits = 0)
intToUtf8(predictors_ridge)

####################################################
###################### LASSO #######################
####################################################
# use the argument alpha = 1 to perform lasso
lasso.mod = glmnet(x[train,], y[train], alpha=1, lambda=grid)
dev.new()
plot(lasso.mod,label = T)
dev.new()
plot(lasso.mod,label = T, xvar = "lambda")
dev.new(); plot_glmnet(lasso.mod, xvar = "lambda")

# perform cross-validation
cv.out=cv.glmnet(x[train,],y[train], alpha=1)
dev.new()
plot(cv.out)
bestlam=cv.out$lambda.min; print(bestlam);print(log(bestlam))
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))

# fit model using glmnet
fit = cv.glmnet(x[train,], y[train], alpha = 1)

# plot cross-validated MSE versus log(lambda)
plot(fit)

lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
mse_lasso <- mean((lasso.pred-y.test)^2); mse_lasso

# wrt lm
lasso.pred=predict(lasso.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
mse_lm <- mean((lasso.pred-y.test)^2); mse_lm

# However, the lasso has a substantial advantage:
# some of the 19 coefficient estimates are exactly zero (12 on the text).
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:51,]
lasso.coef[lasso.coef!=0]
cat("Number of coefficients equal to 0:",sum(lasso.coef==0),"\n")

####################################################
################### Print Results ##################
####################################################
# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save, based on the previous parameters
top_n <- which(lasso.coef >= sort(lasso.coef, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_lasso <- round(lasso.coef[top_n]/100, digits = 0)
intToUtf8(predictors_lasso)


####################################################
################### Elastic Net ####################
####################################################
#for (alp in seq(0.1,0.9,by=0.1)) {
alp <- 0.5
enet.mod = glmnet(x[train,], y[train], alpha=alp, lambda=grid)
dev.new()
plot(enet.mod,label = T)
dev.new()
plot(enet.mod,label = T, xvar = "lambda")
plot_glmnet(enet.mod, xvar = "lambda")
# CV
cv.out=cv.glmnet(x[train,], y[train],alpha=alp,nfolds = 5)
dev.new()
plot(cv.out)
bestlam_enet=cv.out$lambda.min; print(bestlam_enet);print(log(bestlam_enet))
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))
enet.pred=predict(enet.mod,s=bestlam_enet ,newx=x[test,])
mse_enet <- mean((enet.pred-y.test)^2)
print(mse_enet)
#}
elasticNet.coeff =predict(enet.mod,type="coefficients",s=bestlam_enet)[1:51,]


####################################################
################### Print Results ##################
####################################################

# Given the evidence taken from the values in the summary, we can arrive at the number of most important predictors, which is 17.
# In particular, the predictors with bigger t-value and smaller Pr(>|t|) were choosen.
n <- 17 # number of top predictors you want to save, based on the previous parameters
top_n <- which(elasticNet.coeff >= sort(elasticNet.coeff, decreasing = TRUE)[n])
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
predictors_elasticNet <- round(elasticNet.coeff[top_n]/100, digits = 0)
intToUtf8(predictors_elasticNet)
