##### ANALISI ######
startmod < - lm(Y ~ 1, data=data[train, ])
scopmod < - lm(Y ~ ., data[train, ])

optmodAIC < - step(startmod, direction="both", scope=formula(lm(Y ~ ., data[train, ])))
extractAIC(optmodAIC)

optmodBIC < - step(startmod, direction="both",
                   scope=formula(scopmod), k=log(n))  # BIC
extractAIC(optmodBIC, k=log(n))

summ_reg_bwd < - summary(bwd.regfit)
cat("Location of Cp min: ", which.min(bwd.regfit$cp), "\n Coefficients:\n")
print(coef(bwd.regfit, which.min(bwd.regfit$cp)))
cat("Coefficients of optmodAIC:\n")
coefficients(optmodAIC)


###################
intercept_only < - lm(y ~ 1, data=data[train])

optmodAIC < - step(intercept_only, direction="both", scope=formula(lm(Y ~ ., data[train])))
extractAIC()

##### plot ######
plot(bwd.regfit, scale="Cp")
summ_reg_bwd < - summary(bwd.regfit)
cat("Location of Cp min: ", which.min(summ_reg_bwd$cp), "\n Coefficients:\n")
print(coef(regfit.bwd, which.min(summ_reg_bwd$cp)))
cat("Coefficients of optmodAIC:\n")
aic_coeffs < - coefficients(optmodAIC)
print(aic_coeffs)
length(aic_coeffs)
