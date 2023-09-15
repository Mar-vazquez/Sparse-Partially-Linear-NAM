# Real-world datasets survival (Lasso)

## Description:

### Fit the Lasso model to the realsurvival data and
### obtain the C-index, IBS and the features belonging to each structure category

###-----------------------------------------------------------------------------

### Call libraries

library(glmnet)
library(SurvMetrics)

###-----------------------------------------------------------------------------
### Function to obtain survival probabilities of the test individuals at test times
breslow_surv_prob2 <- function(time_interest, pred_test, pred_train, test, train){
  h0 <- c()
  for (t in time_interest){
    ind <- which(train$time <= t)
    dj <- train$status[ind]
    denj <- c()
    for (i in ind){
      ri_ind <- which(train$time >= train$time[i])
      denj <- c(denj, sum(exp(pred_train[ri_ind])))
    }
    h0 <- c(h0, sum(dj/denj))
  }
  
  pred_mat <- matrix(nrow = nrow(test), ncol = length(time_interest))
  for (i in 1:length(time_interest)){
    h0_i = h0[i]
    pred_i= exp(-exp(pred_test) * h0_i) 
    pred_mat[, i] <- pred_i
  }
  return(pred_mat)
}

###-----------------------------------------------------------------------------
### Obtain the predictions for Lasso for all the datasets

cindex_lasso <- c()
IBS_lasso <- c()
struct_lasso <- c()

for (i in 0:4){
  ###-----------------------------------------------------------------------------
  ### Set seed
  set.seed(i)

  ###-----------------------------------------------------------------------------
  ### Read the data
  name_train <- paste("flc_train_",i, ".csv", sep = "")
  name_test <- paste("flc_test_",i, ".csv", sep = "")
  data_train <- read.csv(name_train)[2:12]
  data_test <- read.csv(name_test)[2:12]
  colnames(data_train) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "time1", "time", "status")
  colnames(data_test) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "time1", "time", "status")
  n <- ncol(data_train)
  
  ### Define the time of interest and the survival object
  time_interest <- unique(sort(data_test$time[data_test$time <= quantile(data_test$time, 0.7)]))
  surv_obj <- Surv(data_test$time, data_test$status)
  
  ###-----------------------------------------------------------------------------
  ### Lasso
  lasso.cv <- cv.glmnet(as.matrix(data_train[,1:(n-3)]), as.matrix(data_train[,(n-1):n]), family = "cox")
  lambd_lasso <- lasso.cv$lambda.1se
  fit_lasso <- glmnet(as.matrix(data_train[,1:(n-3)]), as.matrix(data_train[,(n-1):n]), lambda = lambd_lasso, family = "cox")
  coef_lasso <- coef(fit_lasso)[-1]
  pred_lasso <- predict(fit_lasso,  newx = as.matrix(data_test[,1:(n-3)]))
  pred_train <- predict(fit_lasso,  newx = as.matrix(data_train[,1:(n-3)]))
  # Predict survival probabilities of test individuals
  surv_lasso <- breslow_surv_prob2(time_interest, pred_lasso, pred_train, data_test, data_train)
  # IBS
  IBS_lasso <- c(IBS_lasso, IBS(surv_obj, surv_lasso, time_interest))
  # C-index
  cindex_lasso <- c(cindex_lasso, apply(pred_lasso, 2, Cindex, y=as.matrix(data_test[,(n-1):n])))
  # Structure
  sparse_lasso <- which(coef_lasso == 0)
  lin_lasso <- which(coef_lasso != 0)
  non_lin_lasso <- c()
  time_lasso <- c()
  struct_lasso <- c(struct_lasso, list(sparse_lasso, lin_lasso, non_lin_lasso, time_lasso))
  print(i)
}
###-----------------------------------------------------------------------------
### Save results
saveRDS(struct_lasso, file="struct_lasso_flchain.RData")

cindex_lasso <- data.frame(cindex_lasso)
IBS_lasso <- data.frame(IBS_lasso)

write.csv(cindex_lasso, "cindex_lasso_flchain.csv")
write.csv(IBS_lasso, "IBS_lasso_flchain.csv")

###-----------------------------------------------------------------------------
### Obtain mean and sd of the number of features belonging to each structure
sparse_lasso <- c()
lin_lasso <- c()
non_lin_lasso <- c()
for (i in 1:5){
  sparse_lasso <- c(sparse_lasso, length(struct_lasso[[4*(i-1) + 1]]))
  lin_lasso <- c(lin_lasso, length(struct_lasso[[4*(i-1) + 2]]))
  non_lin_lasso <- c(non_lin_lasso, length(struct_lasso[[4*(i-1) + 3]]))
  time_lasso <- c(time_lasso, length(struct_lasso[[4*(i-1) + 4]]))
}

mean(sparse_lasso)
sd(sparse_lasso)

mean(lin_lasso)
sd(lin_lasso)


