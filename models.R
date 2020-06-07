rm(list = ls())

library(forecast)
library(caret)
library(dplyr)
library(tidyr)
library(lubridate)
library(neuralnet)
library(randomForest)
library(nnet)
set.seed(814)

setwd("C:/Users/labbi/OneDrive/Liz/Brandeis/Big data II/assigment data")
weather <- read.csv("weather_hourly_darksky_v2.csv")
weather$Month <- as.factor(weather$Month)
dummy <- as.data.frame(predict(dummyVars(~precipType+Month+icon, data = weather), newdata = weather))
colnames(dummy)[18] <- 'icon.partlycloudy'
dummy <- dummy[,-c(2,14)]
weather_dummy <- cbind(dummy, weather[,c(1:7,9)])

#removw outliers
library (dbscan)
dbscanResult <- dbscan(weather_dummy, eps= 10, minPts=50)
dbscanResult
outliers_index <- which(dbscanResult$cluster == 0)
weather_dummy <- weather_dummy[-which(rownames(weather_dummy) %in% outliers_index), ]
weather <- weather[-which(rownames(weather) %in% outliers_index), ]

#train-valid split
train.index <- sample(1:nrow(weather), round(nrow(weather)*0.8))  
train <- weather[train.index, ]
valid <- weather[-train.index, ]

#logit
logit <- multinom(icon~., data = train)
logit_result <- summary(logit)
round(logit_result$coefficients,2)
pred <- predict(logit, valid)
conf <- confusionMatrix(pred, as.factor(valid$icon))
conf$table
weather_dummy <- weather_dummy[-which(rownames(weather_dummy) %in% outliers_index), ]

pred_train <- predict(logit, train)
confusionMatrix(pred_train, as.factor(train$icon))
logit1 <- glm(icon.fog~.-icon.clear-icon.cloudy-icon.partlycloudy-icon.wind,data=weather_dummy, family='binomial')
par(mfrow = c(2,2))
plot(logit)

#KNN
normalize     = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

weather_norm = as.data.frame(lapply(weather_dummy, normalize))
colnames(weather_norm)
weather_norm <- weather_norm[,-c(13:17)]

train_norm <- weather_norm[train.index, ]
valid_norm <- weather_norm[-train.index, ]
train_y <- train$icon
valid_y <- valid$icon
knnFit1 <- train(train_norm, train_y,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))
knn_results <- knnFit1$results

ggplot(aes(x=k,y=Accuracy),data = knn_results)+geom_line()+theme_bw()+
  labs(title = 'cross-validation score among different k')

KNN_model_pred <- class::knn(train = train_norm, 
                    cl    = train_y,
                    test  = valid_norm,
                    k     = 5)
confusionMatrix(KNN_model_pred, as.factor(valid$icon))

KNN_model_pred_train <- class::knn(train = train_norm, 
                             cl    = train_y,
                             test  = train_norm,
                             k     = 5)
confusionMatrix(KNN_model_pred_train, as.factor(train$icon))

#Linear SVM
train_norm <- cbind(train_norm, train_y)
colnames(train_norm)
library(e1071) 
svm_linear_model = svm(formula = train_y ~ ., data = train_norm, 
                type = 'C-classification', kernel = 'linear',
                cost=1,gamma=0.1) 
svm_pred = predict(svm_linear_model, valid_norm)
confusionMatrix(svm_pred, as.factor(valid$icon))
svm_pred_train = predict(svm_linear_model, train_norm)
confusionMatrix(svm_pred_train, as.factor(train$icon))

#RBF SVM
svm_model = svm(formula = train_y ~ ., data = train_norm, 
                 type = 'C-classification', kernel = 'radial',
                cost=10,gamma=1) 
svm_pred = predict(svm_model, valid_norm)
confusionMatrix(svm_pred, as.factor(valid$icon))
tune.out <- tune(svm, train_y~., data = train_norm, kernel="radial", 
                 ranges = list(gamma=c(0.01,0.1,1,10), 
                               cost = c(0.1,1,10,100)
                 ))

svm_model = svm(formula = train_y ~ ., data = train_norm, 
                type = 'C-classification', kernel = 'radial',
                cost=10,gamma=1) 
svm_pred_train = predict(svm_model, train_norm)
confusionMatrix(svm_pred_train, as.factor(train$icon))

#Random Forest
rf_model <- randomForest(icon~.,data=train,na.action=na.roughfix)
pred_rf  <- predict(rf_model, valid)
confusionMatrix(pred_rf, as.factor(valid$icon))
pred_rf_train<- predict(rf_model, train)
confusionMatrix(pred_rf_train, as.factor(train$icon))

features_gini <- rf_model$importance
features <- rownames(features_gini)
features_gini <- cbind(features,features_gini)
features_gini <- as.data.frame(features_gini)
features_gini$MeanDecreaseGini <- as.numeric(as.character(features_gini$MeanDecreaseGini))
str(features_gini)
library(forcats)
features_gini <- features_gini %>%
  mutate(features = fct_reorder(features,MeanDecreaseGini,last))
ggplot(features_gini, aes(x=features, y = MeanDecreaseGini))+
  geom_col(fill="skyblue", color="black")+
  labs(x = "Mean Decrease Gini",
       y = "Features",
       title = "Features importance")+
  coord_flip()+theme_minimal()

ntree<- 1:100
acc <- vector()
for (i in 1:length(ntree)){
  rf_model <- randomForest(icon~.,data=train,na.action=na.roughfix,ntree=ntree[i])
  pred_rf  <- predict(rf_model, valid)
  table(pred_rf, valid_norm$icon)
  agreement = pred_rf == valid_norm$icon
  acc[i] <- prop.table(table(agreement))[2]
  print(acc[i])
}
df_acc = as.data.frame(cbind(ntree,acc))
ggplot(df_acc,aes(x=ntree,y=acc))+geom_line()+theme_bw()+labs(
  title="Robust Random Tree", x="Number of tree",y="Accuracy"
)

#NB
library(naivebayes)
library(caret)
NBclassifier=naivebayes::naive_bayes(formula   = train_y~.,
                                     usekernel = T,
                                     data      = train_norm)
NB_pred = predict(NBclassifier,valid_norm)
confusionMatrix(NB_pred, as.factor(valid$icon))

#NN
softmax        = function(x) log(1 + exp(x))
relu           = function(x) max(x,0)
NN_model = neuralnet(formula   =  train_y~.,
                       data    = train_norm,
                       hidden  = c(5,2),
                       act.fct = c("logistic",softmax),
                       err.fct = c("sse","ce")[1],
                       threshold = 0.05,
                       stepmax=1e6,
                       rep = 5
)
predict <- compute(NN_model, valid_norm)
NN_pred=apply(predict$net.result,1,which.max)
confusionMatrix(factor(NN_pred,levels=1:5,labels=levels(as.factor(valid_y))), as.factor(valid$icon))

#XGBoost
train.index <- sample(1:nrow(weather), round(nrow(weather)*0.8))  
train <- weather[train.index, ]
valid <- weather[-train.index, ]
train_y <- train$icon
valid_y <- valid$icon
library(data.table)
setDT(train) 
setDT(valid)
train <- model.matrix(~.+0,data = train[,-c("icon"),with=F]) 
valid <- model.matrix(~.+0,data = valid[,-c("icon"),with=F])
train_y <- as.numeric(train_y)-1
valid_y <- as.numeric(valid_y)-1
library(xgboost)
dtrain <- xgb.DMatrix(data = train,label = train_y)
dtest <- xgb.DMatrix(data = valid,label=valid_y)

params <- list(booster = "gbtree", objective = "multi:softprob", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
cv_model <- xgb.cv(params = xgb_params,
                   data = dtrain, 
                   nrounds = 50,
                   nfold = 5,
                   verbose = FALSE,
                   prediction = TRUE)

xgb <- xgboost(data = dtrain, 
               label = train_y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 5,
               nthread = 3
)
prediction <- predict(xgb, dtest)
xgb_pred <- matrix(prediction, nrow = 5,ncol=length(xgb_pred)/5) %>%
  t() %>%
  data.frame() %>%
  mutate(label = as.numeric(valid_y)+1,
         max_prob = max.col(., "last"))
valid_pred<- factor(xgb_pred$max_prob,levels=1:5,labels=levels(as.factor(valid_y)))
confusionMatrix(valid_pred,factor(valid_y))

xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = 5,
                   "label" = train_y)
bst_model <- xgb.train(params = xgb_params,
                       data = dtrain,
                       nrounds = 100)

names <-  colnames(train)
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = xgb)
head(importance_matrix)
library(Ckmeans.1d.dp)
gp = xgb.ggplot.importance(importance_matrix)
print(gp) 

prediction <- predict(bst_model, dtest)
xgb_pred <- matrix(prediction, nrow = 5,ncol=length(xgb_pred)/5) %>%
  t() %>%
  data.frame() %>%
  mutate(label = as.numeric(valid_y)+1,
         max_prob = max.col(., "last"))
dta_valid <- weather[-train.index, ]
valid_pred<- factor(xgb_pred$max_prob,levels=1:5,labels=levels(as.factor(dta_valid$icon)))
confusionMatrix(valid_pred,factor(dta_valid$icon))

dta_train <- weather[train.index, ]
prediction_train <- predict(bst_model, dtrain)
xgb_pred_train <- matrix(prediction_train, nrow = 5,ncol=length(prediction_train)/5) %>%
  t() %>%
  data.frame() %>%
  mutate(label = as.numeric(train_y)+1,
         max_prob = max.col(., "last"))
train_pred<- factor(xgb_pred_train$max_prob,levels=1:5,labels=levels(as.factor(dta_train$icon)))
confusionMatrix(train_pred,factor(dta_train$icon))