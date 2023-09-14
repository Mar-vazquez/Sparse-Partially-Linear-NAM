# Simulation survival (Lasso and Relgam)

## Description:

### Fit the Lasso and Relgam models to the simulated survival data and
### obtain the evaluation metrics (C-index, IBS and wrong structure proportion)

###-----------------------------------------------------------------------------

### Call libraries
library(glmnet)
library(relgam)
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
###-----------------------------------------------------------------------------

### Sequence of proportions, seeds and read ind and elem files 
### This is an example for n = 2000 

seq_prop <- c(1900, 3700, 4600, 5500, 6400, 2080, 3070, 4060, 5050, 7030, 1810, 2620, 3430, 1360, 5230, 1009, 2008, 3007, 5005, 7003, 1711, 2521, 2332, 1261, 2224)
seed_list <- seq(1, 2000)
ind <- read.csv("ind_list.csv")[,2:11]
elem <- read.csv("elem_list.csv")[,2:5]
final_cindex_lasso <- matrix(nrow = 5, ncol = 5)
final_cindex_rel <- matrix(nrow = 15, ncol = 5)

final_IBS_lasso <- matrix(nrow = 5, ncol = 5)
final_IBS_rel <- matrix(nrow = 15, ncol = 5)

final_wrong_lasso <- matrix(nrow = 5, ncol = 5)
final_wrong_rel <- matrix(nrow = 15, ncol = 5)

###-----------------------------------------------------------------------------
### Read data
for (i in 1:15){
  name_train <- paste("train_", seq_prop[i], ".csv", sep = "")
  name_test <- paste("test_", seq_prop[i], ".csv", sep = "") 
  data_train <- read.csv(name_train)[,2:14]
  data_test <- read.csv(name_test)[,2:14]
  colnames(data_train) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "time1", "time", "status")
  colnames(data_test) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "time1", "time", "status")
  
  seed <- seed_list[(5*(i-1) + 1):(5*i)]
  
  ###-----------------------------------------------------------------------------
  
  ### Initialization quantities
  cindex_lasso <- c()
  cindex_rel <- c()
  IBS_lasso <- c()
  IBS_rel <- c()
  struct_lasso <- c()
  struct_rel <- c()

  true_sparse <- ind[i, 1:elem[i,1]] + 1
  if (0 <  elem[i, 2]){
    true_lin <- ind[i, (elem[i, 1] + 1): (elem[i, 1] + elem[i, 2])] + 1
  } else{
    true_lin <- c()
  }
  if (elem[i, 3]> 0){
    true_non_lin <- ind[i, (elem[i, 1] + elem[i, 2] + 1):(elem[i, 1] + elem[i, 2] + elem[i, 3])] + 1
  } else{
    true_non_lin <- c()
  }
  if (elem[i, 4] > 0){
    true_time <- ind[i, (elem[i, 1] + elem[i, 2] + elem[i, 3] + 1): ncol(ind)] + 1
  } else{
    true_time <- c()
  }
  
  time_interest <- unique(sort(data_test$time[data_test$time <= quantile(data_test$time, 0.7)]))
  surv_obj <- Surv(data_test$time, data_test$status)
  for (s in seed){
    set.seed(s)  
    ###-----------------------------------------------------------------------------
    ### Lasso (for the linear)
    if (elem[i, 3] == 0 & elem[i, 4] == 0){
      lasso.cv <- cv.glmnet(as.matrix(data_train[,1:10]), as.matrix(data_train[,12:13]), family = "cox")
      lambd_lasso <- lasso.cv$lambda.1se
      fit_lasso <- glmnet(as.matrix(data_train[,1:10]), as.matrix(data_train[,12:13]), lambda = lambd_lasso, family = "cox")
      coef_lasso <- coef(fit_lasso)
      pred_lasso <- predict(fit_lasso,  newx = as.matrix(data_test[,1:10]))
      pred_train <- predict(fit_lasso,  newx = as.matrix(data_train[,1:10]))
      surv_lasso <- breslow_surv_prob2(time_interest, pred_lasso, pred_train, data_test, data_train)
      IBS_lasso <- c(IBS_lasso, IBS(surv_obj, surv_lasso, time_interest))
      cindex_lasso <- c(cindex_lasso, apply(pred_lasso, 2, Cindex, y=as.matrix(data_test[,12:13])))
      
      sparse_lasso <- which(coef_lasso == 0)
      lin_lasso <- which(coef_lasso != 0)
      non_lin_lasso <- c()
      time_lasso <- c()
      
      wrong_lasso <- 0
      for (l in sparse_lasso){
        if (!(l %in% true_sparse)){
          wrong_lasso <- wrong_lasso + 1
        }
      }
      for (l in lin_lasso){
        if (!(l %in% true_lin)){
          wrong_lasso <- wrong_lasso + 1
        }
      }
      for (l in non_lin_lasso){
        if (!(l %in% true_non_lin)){
          wrong_lasso <- wrong_lasso + 1
        }
      }
      for (l in time_lasso){
        if (!(l %in% true_time)){
          wrong_lasso <- wrong_lasso + 1
        }
      }
      wrong_lasso <- wrong_lasso / 10 
      struct_lasso <- c(struct_lasso, wrong_lasso)
    }
    ###-----------------------------------------------------------------------------
    ### Relgam
    
    if (elem[i, 4] == 0){
      cvfit <- cv.rgam(as.matrix(data_train[,1:10]), as.matrix(data_train[,12:13]),family = "cox")
      lambd <- cvfit$lambda.1se
      fit_rel <- rgam(as.matrix(data_train[,1:10]), as.matrix(data_train[,12:13]), lambda = lambd, family = "cox")
      pred_rel <- predict(fit_rel,  xnew = as.matrix(data_test[,1:10]))
      pred_train <- predict(fit_rel,  xnew = as.matrix(data_train[,1:10]))
      cindex_rel <- c(cindex_rel, apply(pred_rel, 2, Cindex, y=as.matrix(data_test[,12:13])))
      surv_rel <- breslow_surv_prob2(time_interest, pred_rel, pred_train, data_test, data_train)
      IBS_rel <- c(IBS_rel, IBS(surv_obj, surv_rel, time_interest))
      
      nonzero_rel <- fit_rel$feat[[1]]
      non_lin_rel <- fit_rel$nonlinfeat$which
      lin_rel <- nonzero_rel[!nonzero_rel %in% non_lin_rel]
      time_rel <- c()
      total_feat <- seq(1,10)
      sparse_rel <- total_feat[-c(lin_rel, non_lin_rel)]
      
      wrong_rel <- 0
      for (l in sparse_rel){
        if (!(l %in% true_sparse)){
          wrong_rel <- wrong_rel + 1
        }
      }
      for (l in lin_rel){
        if (!(l %in% true_lin)){
          wrong_rel <- wrong_rel + 1
        }
      }
      for (l in non_lin_rel){
        if (!(l %in% true_non_lin)){
          wrong_rel <- wrong_rel + 1
        }
      }
      for (l in time_rel){
        if (!(l %in% true_time)){
          wrong_rel <- wrong_rel + 1
        }
      }
      wrong_rel <- wrong_rel / 10 
      
      struct_rel <- c(struct_rel, wrong_rel)
    }
  }
    ###-----------------------------------------------------------------------------
  if (elem[i, 4] == 0){
    final_IBS_rel[i, ] <- IBS_rel
    final_cindex_rel[i, ] <- cindex_rel
    final_wrong_rel[i, ] <- struct_rel
  }

  if (elem[i, 3] == 0 & elem[i, 4] == 0){
    final_IBS_lasso[i, ] <- IBS_lasso
    final_cindex_lasso[i,] <- cindex_lasso
    final_wrong_lasso[i,] <- struct_lasso
  }
}
###-----------------------------------------------------------------------------

### Save final results
final_wrong_rel <- data.frame(final_wrong_rel)
final_wrong_lasso <- data.frame(final_wrong_lasso)

final_cindex_rel <- data.frame(final_cindex_rel)
final_cindex_lasso <- data.frame(final_cindex_lasso)

final_IBS_rel <- data.frame(final_IBS_rel)
final_IBS_lasso <- data.frame(final_IBS_lasso)

write.csv(final_wrong_rel, "wrong_rel_2000.csv")
write.csv(final_wrong_lasso, "wrong_lasso_2000.csv")


write.csv(final_cindex_rel, "cindex_rel_2000.csv")
write.csv(final_cindex_lasso, "cindex_lasso_2000.csv")

write.csv(final_IBS_rel, "ibs_rel_2000.csv")
write.csv(final_IBS_lasso, "ibs_lasso_2000.csv")