# Real-world datasets classification (Lasso)

## Description:

### Fit the Lasso model to the real binary classification data and
### obtain the F1 score, AUC and the features belonging to each structure category

###-----------------------------------------------------------------------------

### Call libraries
library(glmnet)
library(caret)

### Obtain the predictions for Lasso, Relgam and Gamsel for all the datasets

f1_lasso <- c()
struct_lasso <- c()
auc_lasso <- c()

for (i in 0:4){
  ###-----------------------------------------------------------------------------
  ### Set seed
  set.seed(i)
  
  ###-----------------------------------------------------------------------------
  ### Read the data
  name_train <- paste("titanic_train_",i, ".csv", sep = "")
  name_test <- paste("titanic_test_",i, ".csv", sep = "")
  data_train <- read.csv(name_train)[2:7]
  data_test <- read.csv(name_test)[2:7]
  colnames(data_train) <- c("x1", "x2", "x3", "x4", "x5", "y")
  colnames(data_test) <- c("x1", "x2", "x3", "x4", "x5", "y")
  n <- ncol(data_train)
  
  ###-----------------------------------------------------------------------------
  ### Lasso
  lasso.cv <- cv.glmnet(as.matrix(data_train[,1:(n-1)]), data_train[,n], family = "binomial")
  lambd_lasso <- lasso.cv$lambda.1se
  fit_lasso <- glmnet(as.matrix(data_train[,1:(n-1)]), data_train[,n], lambda = lambd_lasso, family = "binomial")
  coef_lasso <- coef(fit_lasso)[-1]
  pred_lasso <- predict(fit_lasso,  newx = as.matrix(data_test[,1:(n-1)]), type = "response")
  class_pred_lasso <- (pred_lasso > 0.5)*1
  # F1 score
  f1_lasso <- c(f1_lasso, confusionMatrix( as.factor(data_test$y), factor(c(class_pred_lasso), levels = c(0,1)), mode = "everything", positive="1")$byClass[7])
  # AUC
  roc_object <- roc( data_test$y, c(pred_lasso))
  auc_lasso <- c(auc_lasso, auc(roc_object))
  # Structure
  sparse_lasso <- which(coef_lasso == 0)
  lin_lasso <- which(coef_lasso != 0)
  non_lin_lasso <- c()
  struct_lasso <- c(struct_lasso, list(sparse_lasso, lin_lasso, non_lin_lasso))
}

###-----------------------------------------------------------------------------
### Save results
saveRDS(struct_lasso, file="struct_lasso_titanic.RData")

f1_lasso <- data.frame(f1_lasso)
auc_lasso <- data.frame(auc_lasso)

write.csv(f1_lasso, "f1_lasso_titanic.csv")
write.csv(auc_lasso, "auc_lasso_titanic.csv")

###-----------------------------------------------------------------------------
### Obtain mean and sd of the results

# F1 score
mean(as.matrix(f1_lasso))
sd(as.matrix(f1_lasso))

# AUC
mean(as.matrix(auc_lasso))
sd(as.matrix(auc_lasso))

# Structure
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
