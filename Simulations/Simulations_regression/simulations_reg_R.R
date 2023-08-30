### Simulation regression

### Call libraries
library(glmnet)
library(relgam)
library(gamsel)


### Sequence of proportions, seeds and read ind and elem files
#seq_prop <- c(190, 280, 460, 550, 730, 109, 208, 406, 505, 604, 172, 262, 343, 226, 451)
seq_prop <- c(280, 370, 460, 550, 640, 208, 307, 406, 505, 703, 181, 262, 343, 136, 523)
seed_list <- seq(1, 2000)
ind <- read.csv("ind_list.csv")[,2:11]
elem <- read.csv("elem_list.csv")[,2:4]

### Initialize matrices of results
final_mse_lasso <- matrix(nrow = 5, ncol = 5)
final_mse_rel <- matrix(nrow = 15, ncol = 5)
final_mse_gamsel <- matrix(nrow = 15, ncol = 5)

final_wrong_lasso <- matrix(nrow = 5, ncol = 5)
final_wrong_rel <- matrix(nrow = 15, ncol = 5)
final_wrong_gamsel <- matrix(nrow = 15, ncol = 5)

###-----------------------------------------------------------------------------

for (i in 1:15){
  
  ### Read data for train and test
  name_train <- paste("train_", seq_prop[i], ".csv", sep = "")
  name_test <- paste("test_", seq_prop[i], ".csv", sep = "") 
  data_train <- read.csv(name_train)[,2:12]
  data_test <- read.csv(name_test)[,2:12]
  colnames(data_train) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "y")
  colnames(data_test) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "y")
  
  seed <- seed_list[(5*(i-1) + 1):(5*i)]
  
  ###-----------------------------------------------------------------------------
  
  ### Initialization quantities
  mse_lasso <- c()
  mse_rel <- c()
  mse_gamsel <- c()
  struct_lasso <- c()
  struct_rel <- c()
  struct_gamsel <- c()
  
  ### Obtain true structure of features
  true_sparse <- ind[i, 1:elem[i,1]] + 1
  if (elem[i, 2] > 0){
    true_lin <- ind[i, (elem[i, 1] + 1): (elem[i, 1] + elem[i, 2])] + 1
  } else{
    true_lin <- c()
  }
  if (elem[i, 3] > 0){
    true_non_lin <- ind[i, (elem[i, 1] + elem[i, 2] + 1):ncol(ind)] + 1
  } else{
    true_non_lin <- c()
  }
  
  for (s in seed){
    set.seed(s)  
    ###-----------------------------------------------------------------------------
    ### Lasso (for the linear)
    if (elem[i, 3] == 0){
      lasso.cv <- cv.glmnet(as.matrix(data_train[,1:10]), data_train[,11])
      lambd_lasso <- lasso.cv$lambda.1se
      fit_lasso <- glmnet(as.matrix(data_train[,1:10]), data_train[,11], lambda = lambd_lasso)
      coef_lasso <- coef(fit_lasso)[-1]
      pred_lasso <- predict(fit_lasso,  newx = as.matrix(data_test[,1:10]))
      
      ### MSE lasso
      mse_lasso <- c(mse_lasso, mean((data_test$y - pred_lasso)^2))
      
      sparse_lasso <- which(coef_lasso == 0)
      lin_lasso <- which(coef_lasso != 0)
      non_lin_lasso <- c()
      
      ### Wrong structure of lasso
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
      wrong_lasso <- wrong_lasso / 10 
      struct_lasso <- c(struct_lasso, wrong_lasso)
    }
    ###-----------------------------------------------------------------------------
    ### Relgam
    
    cvfit <- cv.rgam(as.matrix(data_train[,1:10]), data_train[,11])
    lambd <- cvfit$lambda.1se
    fit_rel <- rgam(as.matrix(data_train[,1:10]), data_train[,11], lambda = lambd)
    pred_rel <- predict(fit_rel,  xnew = as.matrix(data_test[,1:10]))
    
    ### MSE relgam
    mse_rel <- c(mse_rel, mean((data_test$y - pred_rel)^2))

    nonzero_rel <- fit_rel$feat[[1]]
    non_lin_rel <- fit_rel$nonlinfeat$s0
    lin_rel <- nonzero_rel[!nonzero_rel %in% non_lin_rel]
    total_feat <- seq(1,10)
    sparse_rel <- total_feat[-c(lin_rel, non_lin_rel)]
    
    ### Wrong structure relgam
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
    wrong_rel <- wrong_rel / 10 
    
    struct_rel <- c(struct_rel, wrong_rel)
    
    ###-----------------------------------------------------------------------------
    ### Gamsel
    
    gamsel.cv=cv.gamsel((data_train[,1:10]), data_train$y)
    ind_opt <- gamsel.cv$index.1se
    fit_gamsel <- gamsel((data_train[,1:10]), data_train$y)
    pred_gamsel <- predict(fit_gamsel,  newdata = data_test[,1:10], index = ind_opt)
    
    ### MSE Gamsel
    mse_gamsel <- c(mse_gamsel, mean((data_test$y - pred_gamsel)^2))
    
    getActive(fit_gamsel, index = ind_opt, type = "linear")
    getActive(fit_gamsel, index = ind_opt, type = "nonlinear")
    lopt <- paste("l", ind_opt, sep = "")
    nonzero_gamsel <- getActive(fit_gamsel, index = ind_opt, type = "nonzero")[[lopt]]
    sparse_gamsel <- total_feat[- nonzero_gamsel]
    non_lin_gamsel <- getActive(fit_gamsel, index = ind_opt, type = "nonlinear")[[lopt]]
    lin_gamsel <- nonzero_gamsel[!nonzero_gamsel %in% non_lin_gamsel]
    
    ### Wrong structure Gamsel
    wrong_gamsel <- 0
    for (l in sparse_gamsel){
      if (!(l %in% true_sparse)){
        wrong_gamsel <- wrong_gamsel + 1
      }
    }
    for (l in lin_gamsel){
      if (!(l %in% true_lin)){
        wrong_gamsel <- wrong_gamsel + 1
      }
    }
    for (l in non_lin_gamsel){
      if (!(l %in% true_non_lin)){
        wrong_gamsel <- wrong_gamsel + 1
      }
    }
    wrong_gamsel <- wrong_gamsel / 10 
    struct_gamsel <- c(struct_gamsel, wrong_gamsel)
    }
  
  
  final_mse_rel[i, ] <- mse_rel
  final_mse_gamsel[i, ] <- mse_gamsel
  
  final_wrong_rel[i, ] <- struct_rel
  final_wrong_gamsel[i, ] <- struct_gamsel
  
  if (elem[i, 3] == 0){
    final_mse_lasso[i,] <- mse_lasso
    final_wrong_lasso[i,] <- struct_lasso
  }
}
###-----------------------------------------------------------------------------

### Save final results
final_wrong_gamsel <- data.frame(final_wrong_gamsel)
final_wrong_rel <- data.frame(final_wrong_rel)
final_wrong_lasso <- data.frame(final_wrong_lasso)

final_mse_gamsel <- data.frame(final_mse_gamsel)
final_mse_rel <- data.frame(final_mse_rel)
final_mse_lasso <- data.frame(final_mse_lasso)

write.csv(final_wrong_gamsel, "wrong_gamsel_1000.csv")
write.csv(final_wrong_rel, "wrong_rel_1000.csv")
write.csv(final_wrong_lasso, "wrong_lasso_1000.csv")


write.csv(final_mse_gamsel, "mse_gamsel_1000.csv")
write.csv(final_mse_rel, "mse_rel_1000.csv")
write.csv(final_mse_lasso, "mse_lasso_1000.csv")
