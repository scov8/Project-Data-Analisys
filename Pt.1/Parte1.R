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
library(corrplot)
library(psych)
library(leaps) # per i bss
library(car) # per vif

# Carichiamo il dataset
data <- read.csv("RegressionDataset_DA_group1.csv", header = T, na.strings = "?")

head(data)
# unique(unlist (lapply (Auto, function (x) which (is.na (x))))) #To find all the rows in a data frame with at least one NA
data <- na.omit(data)

# Creo i due set
set.seed(1985) # Utilizziamo un seed fissato per la riprodurre l'esperimento

# Salvo in x e y i valori dell'intero dataset
x <- model.matrix(Y ~ ., data)[, -1] # Salvo in x tutti i valori tranne quelli di Y
y <- data$Y # Salvo in y i valori della colonna Y

train <- sample(1:nrow(x), 0.8 * nrow(x)) # another typical approach to sample
test <- (-train)
y.test <- y[test]

# parameters
n <- nrow(data[train, ])
p <- ncol(x[train, ])
set.seed(1985)


####################################################
################### Studio dati ####################
####################################################

############# MATRICE DI CORRELAZIONE ##############
dev.new()
corData <- round(cor(x), digits = 2)
corPlot(corData, cex = 0.22, show.legend = TRUE, main = "Correlation plot")

####################################################
########## REGRESSIONE LINEARE MULTIPLA ############
####################################################
lm.mod <- lm(Y ~ ., data = data[train, ])
summary(lm.mod)
lm.predict <- predict(lm.mod, newdata = (data[test, ]))
lm.coeff <- coef(lm.mod)
lm.mse <- mean((lm.predict - y[test])^2)
lm.mse

vif(lm.mod) # verifica la collinearità dei dati

####################################################
###################### BSS #########################
####################################################

# Il seguente codice. effettua il BSS, che però computazionalmente non è efficiente, quindi con una p
# circa maggiore di 30 si impiega troppo tempo per la computazione dello stesso
# full.regfit <- regsubsets(Y ~ ., nvmax = variables_selected, data = data[train, ], really.big = T) # BSS
# summary(full.regfit)

####################################################
################### Stepwise #######################
####################################################

################# Forward selection ################
fwd.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "forward")
summary(fwd.regfit)

################# Backward selection ################
# per applicare il backword dobbiamo avere n>p
bwd.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "backward")
summary(bwd.regfit)

################## Hybrid selection ##################
hyb.regfit <- regsubsets(Y ~ ., data = data[train, ], nvmax = p, method = "seqrep")
summary(hyb.regfit)

################### Test ##################
test.mat <- model.matrix(Y ~ ., data = data[test, ])

val.errors <- rep(NA, p)
for (i in 1:p) {
  coefi <- coef(bwd.regfit, id = i)
  pred <- test.mat[, names(coefi)] %*% coefi
  val.errors[i] <- mean((y[test] - pred)^2)
}

# The best model is the one that contains which.min(val.errors) (ten in the book) variables.
min <- which.min(val.errors)
bwd.mse <- val.errors[min]
bwd.coef <- coef(bwd.regfit, min) # This is based on training data

val.errors <- rep(NA, p)
for (i in 1:p) {
  coefi <- coef(fwd.regfit, id = i)
  pred <- test.mat[, names(coefi)] %*% coefi
  val.errors[i] <- mean((y[test] - pred)^2)
}

# The best model is the one that contains which.min(val.errors) (ten in the book) variables.
min <- which.min(val.errors)
fwd.mse <- val.errors[min]
fwd.coef <- coef(fwd.regfit, which.min(val.errors)) # This is based on training data

val.errors <- rep(NA, p)
for (i in 1:p) {
  coefi <- coef(hyb.regfit, id = i)
  pred <- test.mat[, names(coefi)] %*% coefi
  val.errors[i] <- mean((y[test] - pred)^2)
}

# The best model is the one that contains which.min(val.errors) (ten in the book) variables.
min <- which.min(val.errors)
hyb.mse <- val.errors[min]
hyb.coef <- coef(hyb.regfit, which.min(val.errors)) # This is based on training data

################### Print Results ##################
bwd.predictors <- round(bwd.coef / 100, digits = 0)
intToUtf8(bwd.predictors)

fwd.predictors <- round(fwd.coef / 100, digits = 0)
intToUtf8(fwd.predictors)

hyb.predictors <- round(hyb.coef / 100, digits = 0)
intToUtf8(hyb.predictors)

######################## Plot ########################
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

####################################################
###################### LASSO #######################
####################################################
grid <- 10^seq(5, -2, length = 1000)

# use the argument alpha = 1 to perform lasso
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
dev.new()
plot(lasso.mod, label = T)
# dev.new()
# plot(lasso.mod, label = T, xvar = "lambda") # altro modo per plottare al variare di lambda
dev.new()
plot_glmnet(lasso.mod, xvar = "lambda")

# perform cross-validation
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
dev.new()
plot(cv.out)
bestlam <- cv.out$lambda.1se

lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
lasso.mse <- mean((lasso.pred - y[test])^2)
lasso.mse

# However, the lasso has a substantial advantage:
# some of the 19 coefficient estimates are exactly zero (12 on the text).
lasso.out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(lasso.out, type = "coefficients", s = bestlam)[1:p + 1, ]
lasso.coef
lasso.coef[lasso.coef != 0]
cat("Number of coefficients equal to 0:", sum(lasso.coef == 0), "\n")

# dev.new()
# plot(lasso.out,label = T, xvar = "lambda")  # altro modo per plottare al variare di lambda
dev.new()
plot_glmnet(lasso.out, xvar = "lambda")


################### Print Results ##################
lasso.predictors <- round(lasso.coef / 100, digits = 0)
intToUtf8(lasso.predictors)


####################################################
###################### Ridge #######################
####################################################
# andando a calcolare il lambda sarà sempre vicino allo 0, quindi diventa OLS cioè un modello lineare
grid <- 10^seq(5, -2, length = 1000)

ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid) # si esegue la refressione ridge
dev.new()
plot(ridge.mod, label = T)
# dev.new()
# plot(ridge.mod, label = T, xvar = "lambda")  # altro modo per plottare al variare di lambda
dev.new()
plot_glmnet(ridge.mod, xvar = "lambda")
dim(coef(ridge.mod)) # dimensione dei coefficienti

# perform cross-validation
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0, lambda = grid)
dev.new()
plot(cv.out)
bestlam <- cv.out$lambda.min

ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ])
ridge.mse <- mean((ridge.pred - y[test])^2)
ridge.mse

ridge.out <- glmnet(x, y, alpha = 0, lambda = grid)
ridge.coef <- predict(ridge.out, type = "coefficients", s = bestlam)[1:p + 1, ]
ridge.coef

# dev.new()
# plot(ridge.out,label = T, xvar = "lambda")
dev.new()
plot_glmnet(ridge.out, xvar = "lambda")

################### Print Results ##################
ridge.predictors <- round(ridge.coef / 100, digits = 0)
intToUtf8(ridge.predictors)

####################################################
################### Elastic Net ####################
####################################################
enet.mod <- glmnet(x[train, ], y[train], alpha = 0.1, lambda = grid)
enet.pred <- predict(enet.mod, s = 0.1, newx = x[test, ])
min <- mean((enet.pred - y.test)^2)
def.alp <- 0
for (alp in seq(0.1, 0.9, by = 0.05)) { # siamo più dalla parte di lasso x il fatto che per dati molto incorrelati tra loro non conviene applicare ridge
  enet.mod <- glmnet(x[train, ], y[train], alpha = alp, lambda = grid)
  cv.out <- cv.glmnet(x[train, ], y[train], alpha = alp, lambda = grid)
  enet.bestlam <- cv.out$lambda.min

  enet.pred <- predict(enet.mod, s = enet.bestlam, newx = x[test, ])
  enet.mse <- mean((enet.pred - y.test)^2)

  if (enet.mse < min) {
    min <- enet.mse
    def.alp <- alp
  }
}

enet.mod <- glmnet(x[train, ], y[train], alpha = def.alp, lambda = grid)
dev.new()
plot(enet.mod, label = T)
dev.new()
plot(enet.mod, label = T, xvar = "lambda")
plot_glmnet(enet.mod, xvar = "lambda")
# CV
cv.out <- cv.glmnet(x[train, ], y[train], alpha = def.alp, lambda = grid)
dev.new()
plot(cv.out)
enet.bestlam <- cv.out$lambda.min

enet.pred <- predict(enet.mod, s = enet.bestlam, newx = x[test, ])
enet.mse <- mean((enet.pred - y.test)^2)

enet.coeff <- predict(enet.mod, type = "coefficients", s = enet.bestlam)[1:51, ]

################### Print Results ##################
# Finally, we took the vector with the best predictors estimate, we diveded by 100 and and we make the conversation in ASCII.
enet.predictors <- round(enet.coeff / 100, digits = 0)
intToUtf8(enet.predictors)

####################################################
################### Stampa MSE #####################
####################################################
lm.mse
fwd.mse
bwd.mse
hyb.mse
lasso.mse
ridge.mse
enet.mse
