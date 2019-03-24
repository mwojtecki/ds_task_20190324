#Upload packages and functions

source('packages.r')
source('helpers.r')

#Data preparation

df  <- read.csv('creditcard.csv')
idx <- c(1:nrow(df))
df  <- cbind(idx,df)

time    <- df$Time
time    <- ifelse(time>=86400,time-86400,time)
df$Time <- time

df$Class <- as.factor(df$Class)

sets_list <- train_valid_div(df,0.66,'idx')
train <- sets_list[["train"]]
valid <- sets_list[["valid"]]

train <- train %>% select(-idx,-Amount)
valid <- valid %>% select(-idx,-Amount)

#Random Forest

feat_imp_rf <- randomForest(Class ~ ., data=train)
cutoff <- cutoff_exam(feat_imp_rf,train$Class,0.05,0.95)

opt_th <- cutoff[['opt_th']]

#kNN

imp <- as.data.frame(feat_imp_rf$importance)
imp$var <- rownames(imp)
vars <- (imp %>% arrange(desc(MeanDecreaseGini)) %>% filter(MeanDecreaseGini > 40))$var

cv_df_knn <- subset(df, select=c(vars))
cv_df_knn_y <- subset(df, select=Class)
cv_df_knn <- cbind(cv_df_knn, cv_df_knn_y)

folds <- createFolds(cv_df_knn$Class, k = 10, list = TRUE, returnTrain = FALSE)

results <- cv_knn_train(folds, cv_df_knn, 10)
df_cv <- as.data.frame(t(as.data.frame(results)),row.names = F)

means <- df_cv %>% group_by(k) %>% summarise_all(mean)

knn_train_y <- train$Class
knn_valid_y <- valid$Class

knn_train <- subset(train,select=vars) %>% mutate(Class=knn_train_y)
knn_valid <- subset(valid,select=vars) %>% mutate(Class=knn_valid_y)

valid3nn <- DMwR::kNN(Class~.,knn_train,knn_valid,k=3)
train3nn <- DMwR::kNN(Class~.,knn_train,knn_train,k=3)

valid_t <- table(knn_valid_y,valid3nn)
train_t <- table(knn_train_y,train3nn)

#Amount prediction

fraud <- df %>% filter(Class==1)
legal <- df %>% filter(Class==0)

test       <- rbind(fraud, legal %>% sample_frac(0.3))
test_class <- test$Class
test_y     <- test$Amount

train   <- anti_join(df, test, by='idx')
train_y <- train$Amount

test    <- subset(test, select=c(-Class,-Amount,-idx))
train   <- subset(train, select=c(-Class,-Amount,-idx))

dtrain <- xgb.DMatrix(data=as.matrix(train), label=train_y)
dtest <- xgb.DMatrix(data=as.matrix(test),   label=test_y)

watchlist <- list(train=dtrain, test=dtest)

eta_levels = seq(0.05,0.5,0.05)
dep_levels = c(2:5)

cv_bst <- cv_xgb(eta_lvs = eta_levels,dep_lvs = dep_levels,data = dtrain,n = 5,round = 50)

eval_id <- seq(3,120,3)

test_results <- NULL
for (i in eval_id){
    res <- c(unlist((cv_bst[[i]][50,4])))
    test_results <- c(test_results, res)
}

train_results <- NULL
for (i in eval_id){
    res <- c(unlist((cv_bst[[i]][50,2])))
    train_results <- c(train_results, res)
}

bst <- xgb.train(data=dtrain, max.depth=2, eta=0.5, nthread=3,nround=50,
                 watchlist = watchlist,objective = "reg:linear")

frauds <- test[test_class==1,]
f_preds <- predict(bst,as.matrix(frauds))
f_preds <- ifelse(f_preds < 0, 0, f_preds)

RMSE(test_y[test_class==1],f_preds)

R2(test_y[test_class==1],f_preds)