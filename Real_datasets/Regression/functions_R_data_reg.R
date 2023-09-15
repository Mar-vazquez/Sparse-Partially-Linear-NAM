# Real-world datasets regression (Lasso, Relgam and Gamsel)

## Description:

### Fit the Lasso, Relgam and Gamsel models to the real regression data and
### obtain the MSE and the features belonging to each structure category

###-----------------------------------------------------------------------------

### Call libraries
library(glmnet)
library(relgam)
library(gamsel)

###-----------------------------------------------------------------------------

### Obtain the predictions for Lasso, Relgam and Gamsel for all the datasets

mse_lasso <- c()
mse_rel <- c()
mse_gamsel <- c()
struct_lasso <- c()
struct_rel <- c()
struct_gamsel <- c()

for (i in 0:4){
  ###-----------------------------------------------------------------------------
  ### Set seed
  set.seed(i)
  
  ###-----------------------------------------------------------------------------
  ### Read the data
  name_train <- paste("wine_train_",i, ".csv", sep = "")
  name_test <- paste("wine_test_",i, ".csv", sep = "")
  data_train <- read.csv(name_train)[2:13]
  data_test <- read.csv(name_test)[2:13]
  colnames(data_train) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "y")
  colnames(data_test) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "y")
  n <- ncol(data_train)
  
  ###-----------------------------------------------------------------------------
  ### Lasso
  lasso.cv <- cv.glmnet(as.matrix(data_train[,1:(n-1)]), data_train[,n])
  lambd_lasso <- lasso.cv$lambda.1se
  fit_lasso <- glmnet(as.matrix(data_train[,1:(n-1)]), data_train[,n], lambda = lambd_lasso)
  coef_lasso <- coef(fit_lasso)[-1]
  pred_lasso <- predict(fit_lasso,  newx = as.matrix(data_test[,1:(n-1)]))
  mse_lasso <- c(mse_lasso, mean((data_test$y - pred_lasso)^2))
  
  sparse_lasso <- which(coef_lasso == 0)
  lin_lasso <- which(coef_lasso != 0)
  non_lin_lasso <- c()
  struct_lasso <- c(struct_lasso, list(sparse_lasso, lin_lasso, non_lin_lasso))
  
  ###-----------------------------------------------------------------------------
  ### Relgam
  
  cvfit <- cv.rgam(as.matrix(data_train[,1:(n-1)]), data_train[,n])
  lambd <- cvfit$lambda.1se
  fit_rel <- rgam(as.matrix(data_train[,1:(n-1)]), data_train[,n], lambda = lambd)
  pred_rel <- predict(fit_rel,  xnew = as.matrix(data_test[,1:(n-1)]))
  mse_rel <- c(mse_rel, mean((data_test$y - pred_rel)^2))
  
  nonzero_rel <- fit_rel$feat[[1]]
  non_lin_rel <- fit_rel$nonlinfeat$s0
  lin_rel <- nonzero_rel[!nonzero_rel %in% non_lin_rel]
  total_feat <- seq(1, (n-1))
  sparse_rel <- total_feat[-c(lin_rel, non_lin_rel)]
  struct_rel <- c(struct_rel, list(sparse_rel, lin_rel, non_lin_rel))
  
  ###-----------------------------------------------------------------------------
  ### Gamsel
  gamsel.cv=cv.gamsel((data_train[,1:(n-1)]), data_train$y, degrees = rep(5, n-1))
  ind_opt <- gamsel.cv$index.1se
  fit_gamsel <- gamsel((data_train[,1:(n-1)]), data_train$y, degrees = rep(5, n-1))
  pred_gamsel <- predict(fit_gamsel,  newdata = data_test[,1:(n-1)], index = ind_opt)
  mse_gamsel <- c(mse_gamsel, mean((data_test$y - pred_gamsel)^2))
  
  lopt <- paste("l", ind_opt, sep = "")
  nonzero_gamsel <- getActive(fit_gamsel, index = ind_opt, type = "nonzero")[[lopt]]
  sparse_gamsel <- total_feat[- nonzero_gamsel]
  non_lin_gamsel <- getActive(fit_gamsel, index = ind_opt, type = "nonlinear")[[lopt]]
  lin_gamsel <- nonzero_gamsel[!nonzero_gamsel %in% non_lin_gamsel]
  struct_gamsel <- c(struct_gamsel, list(sparse_gamsel, lin_gamsel, non_lin_gamsel))
}

###-----------------------------------------------------------------------------
### Save results

saveRDS(struct_gamsel, file="struct_gamsel_wine.RData")
saveRDS(struct_rel, file="struct_rel_wine.RData")
saveRDS(struct_lasso, file="struct_lasso_wine.RData")

mse_gamsel <- data.frame(mse_gamsel)
mse_rel <- data.frame(mse_rel)
mse_lasso <- data.frame(mse_lasso)

write.csv(mse_gamsel, "mse_gamsel_wine.csv")
write.csv(mse_rel, "mse_rel_wine.csv")
write.csv(mse_lasso, "mse_lasso_wine.csv")

###-----------------------------------------------------------------------------
### Obtain mean and sd of the results

# MSE Lasso
mean(as.matrix(mse_lasso))
sd(as.matrix(mse_lasso))

# Structure Lasso
sparse_lasso <- c()
lin_lasso <- c()
non_lin_lasso <- c()
for (i in 1:5){
  sparse_lasso <- c(sparse_lasso, length(struct_lasso[[3*(i-1) + 1]]))
  lin_lasso <- c(lin_lasso, length(struct_lasso[[3*(i-1) + 2]]))
  non_lin_lasso <- c(non_lin_lasso, length(struct_lasso[[3*(i-1) + 3]]))
}

mean(sparse_lasso)
sd(sparse_lasso)

mean(lin_lasso)
sd(lin_lasso)


# MSE Relgam
mean(as.matrix(mse_rel))
sd(as.matrix(mse_rel))

# Structure Relgam
sparse_rel <- c()
lin_rel <- c()
non_lin_rel <- c()
for (i in 1:5){
  sparse_rel <- c(sparse_rel, length(struct_rel[[3*(i-1) + 1]]))
  lin_rel <- c(lin_rel, length(struct_rel[[3*(i-1) + 2]]))
  non_lin_rel <- c(non_lin_rel, length(struct_rel[[3*(i-1) + 3]]))
}

mean(sparse_rel)
sd(sparse_rel)

mean(lin_rel)
sd(lin_rel)

mean(non_lin_rel)
sd(non_lin_rel)


# MSE Gamsel
mean(as.matrix(mse_gamsel))
sd(as.matrix(mse_gamsel))

# Structure Gamsel
sparse_gamsel <- c()
lin_gamsel <- c()
non_lin_gamsel <- c()
for (i in 1:5){
  sparse_gamsel <- c(sparse_gamsel, length(struct_gamsel[[3*(i-1) + 1]]))
  lin_gamsel <- c(lin_gamsel, length(struct_gamsel[[3*(i-1) + 2]]))
  non_lin_gamsel <- c(non_lin_gamsel, length(struct_gamsel[[3*(i-1) + 3]]))
}

mean(sparse_gamsel)
sd(sparse_gamsel)

mean(lin_gamsel)
sd(lin_gamsel)

mean(non_lin_gamsel)
sd(non_lin_gamsel)






