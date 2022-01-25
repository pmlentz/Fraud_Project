#load libraries
library(tidyverse)
library(caret)
library(imbalance)
library(earth)
library(nnet)
library(randomForest)
library(xgboost)
library(naivebayes)
library(modEvA)
library(ROCR)
library(precrec)
library(InformationValue)

#Read Data In
data <- read.csv("C:/Users/lentp/Downloads/1056lab-credit-card-fraud-detection/train.csv")
set.seed(1234)
train <- data %>% sample_frac(0.7)
test <- anti_join(data, train, by= "ID")
rm(data)
table(data$Class)
#check columns for NAs
for(i in 1:length(train)){
  print(sum(is.na(train[i])))
}
#None

####Explore distribution of frauds/not frauds####
table(train$Class)
table(train$Class)[2]/(table(train$Class)[1]+table(train$Class)[2])
#.1922855% of transactions classified as fraud
#Clear rare event bias to address

#######################Rare Event Addressing######################
#how many do new positives do we need to fix rare event?

numPositive <- length(which(train$Class == 1))
numNegative <- length(which(train$Class == 0))
n5p <- length(train$Class)/20
n5p-numPositive
length(train$Class)
(6676+numPositive)/length(train$Class)
#need at least 6676 but adding that many we will need to add a few more to stay above 5%
7309/146165
#factoring the 5% into the last 6676 to keep it above 5% with new data
#7309 total positives out of a minimum of 146165 total

#Make the response a factor for ease and remove index column
train$Class <- as.factor(train$Class)
train <-train[-1]
#Lets try two different oversampling techniques, and see which one outperforms
pdos_noclean <- pdfos(train, numInstances = 7309, classAttr = "Class")

smote_noclean <- mwmote(train, numInstances = 7309, classAttr="Class")

pdos_set_noclean <- rbind(train, pdos_noclean)
smote_set_noclean <- rbind(train, smote_noclean)
#visualze the two types of synthetic sets
plotComparison(train[c(2:3,31)], pdos_set_noclean[c(2:3,31)], attrs = names(train)[c(2:3,31)], classAttr="Class")
plotComparison(train[c(2:3,31)], smote_set_noclean[c(2:3,31)], attrs = names(train)[c(2:3,31)], classAttr="Class")
#Smote did a better job time to make more and "clean"
#Create  45000 synthetic positives with smote so we can clean and end up with over 7309
smote_os <- mwmote(train, numInstances = 45000, classAttr="Class")
#This algorithm uses game theory  and knn to try and filter and clean "bad" synthetics
set.seed(1234)
smote_filtered <- neater(train, smote_os, iterations = 100, k=10) 
plottest <- rbind(train, smote_os)
#recombine filtered set
smote_final <-  rbind(train, smote_filtered)
#visualize
plotComparison(train[c(2:3,31)], plottest[c(2:3,31)], attrs = names(train)[c(2:3,31)], classAttr="Class")
plotComparison(train[c(2:3,31)], smote_final[c(2:3,31)], attrs = names(train)[c(2:3,31)], classAttr="Class")


table(smote_final$Class)
9375/(138589+9375)
#over 6% so the rare event is addressed

#create weights to bias correct
weight <- 19
weights <- ifelse(smote_final$Class == 1, 1, weight)

#clean section
rm(smote_filtered, smote_os, smote_set_noclean, pdos_set_noclean, smote_noclean,
   pdos_noclean,n5p, numPositive, numNegative)

#Start ML on combined set!!!

########################Logistic Regression######################
LR_Model_F <- glm(Class~., data = smote_final, family=binomial(link="logit"), weight = weights)
LR_Model_E <- glm(Class~1, data = smote_final, family=binomial(link="logit"), weight = weights)
#FW Selection
LR_Model_Fw <- step(LR_Model_E, scope = list(lower=LR_Model_E,
                                             upper=LR_Model_F),
                    direction="forward", k = log(nrow(smote_final)))
summary(LR_Model_Fw)

#BW Selection
LR_Model_Bw <- step(LR_Model_F, scope = list(lower=LR_Model_E,
                                             upper=LR_Model_F),
                    direction="backward", k = log(nrow(smote_final)))

#Both
LR_Model_Bo <- step(LR_Model_E, scope = list(lower=LR_Model_E,
                                             upper=LR_Model_F),
                    direction="both", k = log(nrow(smote_final)))
#model summarys (Are any different?)
summary(LR_Model_Fw) #10133 AIC ***** Best
summary(LR_Model_Bw) #10135 AIC 
summary(LR_Model_Bo) #10135 AIC
LR_Model <- LR_Model_Fw #just for nomenclature consistency rename

#Score!
lr_predict <- predict(LR_Model, type= 'response')
plotROC(smote_final$Class, lr_predict) #98.53%
lr_eval <- evalmod(scores = lr_predict, labels = smote_final$Class)
auc(lr_eval)
ks_stat(actuals =smote_final$Class, predictedScores = lr_predict) #.9528
lr_predict <- as.numeric(lr_predict > 0.5)
caret::confusionMatrix(as.factor(lr_predict),smote_final$Class)
#Specificity: .9998
#Sensitivity: .9250

#Clean Section
rm(LR_Model_Fw, LR_Model_Bo, LR_Model_E, LR_Model_Bw, LR_Model_F, lr_predict)

###################### Neural Network #########################

#standardize non principal components
nnet_set <- smote_final
nnet_set$Time <- scale(nnet_set$Time)
nnet_set$Amount <- scale(nnet_set$Amount)
#Tune
tune_grid <- expand.grid(
  .size = c(3:7),
  .decay = c(0, 0.5, 1)
)
set.seed(1234)
NN_Model_Tune <- caret::train(Class~., data= nnet_set,
                       method = "nnet",
                       tuneGrid=tune_grid,
                       weights = weights,
                       trControl= trainControl(method = 'cv', number = 10, allowParallel = T)) 
NN_Model_Tune$bestTune
#Size 7 decay 1

NN_Model <- nnet(Class~., data=nnet_set, size = 7, decay = 1, linout = F, weights = weights)

#Score!
nn_predict <- predict(NN_Model, type= 'raw')
plotROC(smote_final$Class, nn_predict) #99.76%
nn_eval <- evalmod(scores = nn_predict, labels = smote_final$Class)
auc(nn_eval) #99.71933%
ks_stat(actuals =smote_final$Class, predictedScores = nn_predict) #.9576
nn_predict <- predict(NN_Model, type= 'class')
caret::confusionMatrix(as.factor(nn_predict),smote_final$Class)
#Specificity: 0.9998
#Sensitivity: 0.9629

#Clean Section
rm(NN_Model_Tune, nn_predict)

##################Random Forest##########################

#create a dataset for RF with a random variable for variable optimization
rf_tune <- smote_final
set.seed(1234)
rf_tune$random <- rnorm(147964)
set.seed(1234)
rf_model_tune <- randomForest(as.factor(Class)~., data = data.frame(rf_tune), ntree=200, importance=T, classwt = c(19,1))
plot(rf_model_tune)
#50 seems like an appropriate amount of trees
importance(rf_model_tune,sort=T)
#all variables are better then the random so lets keep them all!
#since keeping them all just use normal smote_final dataset

#tune mtry
set.seed(1234)
tuneRF(x=smote_final[,-31],y=as.factor(smote_final[,31]),plot=T,ntreeTry = 50,stepFactor = .5, classwt=c(19,1))
#mtry 5 seems best

#build final model
set.seed(1234)
RF_Model <- randomForest(Class~., data=smote_final, ntree=50, mtry=5,importance=T, classwt = c(19,1))

#Score!
rf_predict <- predict(RF_Model, type= 'prob')[,2]
plotROC(smote_final$Class, rf_predict) #.9984
rf_eval <- evalmod(scores = rf_predict, labels = smote_final$Class)
auc(rf_eval) #99.72120%
ks_stat(actuals =smote_final$Class, predictedScores = rf_predict) #.9578
rf_predict <- predict(RF_Model, type= 'response')
caret::confusionMatrix(rf_predict,smote_final$Class)
#Specificity:.9999
#Sensitivity:.9947

#clean section
rm(rf_model_tune, rf_tune)

#######################XGBoost#############################

#format in a way that xgboost wants
trainx <- model.matrix(Class~.,data=smote_final)[,-31]
trainy <- as.numeric(levels(smote_final$Class))[smote_final$Class]
#start tuning
set.seed(1234)
xgb_model_tune <- xgb.cv(data = trainx, label = trainy, subsample =.5, nrounds = 100,
                     objective = "binary:logistic", nfold = 10, weight=weights)
#which nrounds was best:
which.min(xgb_model_tune$evaluation_log$test_logloss_mean)
#45
tune_grid = expand.grid(
  nrounds = 45,
  eta = c(0.2, 0.3, 0.4),
  max_depth = c(8:11),
  gamma = c(0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.75, 1)
)
set.seed(1234)
xgb_model_tune <- train(x=trainx, y =as.factor(trainy),
                   method = "xgbTree",
                   tuneGrid = tune_grid,
                   weights=weights,
                   trControl=trainControl(method = 'cv',
                                          number = 10,
                                          allowParallel = T))
xgb_model_tune$bestTune
plot(xgb_model_tune)
# 1, 10, .4
set.seed(1234)
XGB_Model <- xgboost(data = trainx, label = trainy, 
                   subsample =1, nrounds = 45, 
                   max_depth = 10, eta = .4,
                   objective = "binary:logistic")

#Score!
xgb_predict <- predict(XGB_Model, newdata = trainx, type= 'raw')
plotROC(smote_final$Class, xgb_predict) #100%
xg_eval <- evalmod(scores = xgb_predict, labels = smote_final$Class)
auc(xg_eval)
xgb_predict <- predict(XGB_Model, newdata = trainx, type= 'response')
xgb_predict <- as.numeric(xgb_predict > 0.5)
caret::confusionMatrix(as.factor(xgb_predict),smote_final$Class)
#Specificity: 1 
#Sensitivity: 1
#none wrong

#clean section
rm(xgb_model_tune, trainx, trainy, xgb_predict)

##########################NaiveBayes###########################
tune_grid <- expand.grid(
  fL = c(0,.5, 1),
  usekernel = c(T, F),
  adjust=c(0, .5, 1))

set.seed(1234)
NB_Model_Tune <- train(x=as.data.frame(smote_final[,-31]),
                             y=smote_final$Class,
                             method = "nb",
                             weights = weights,
                             tuneGrid = tune_grid,
                             trControl = trainControl(method = 'cv',
                                                      number=10,
                                                      allowParallel = T))
NB_Model_Tune$bestTune
plot(NB_Model_Tune)
set.seed(1234)

NB_Model <- naive_bayes(Class ~ ., data = smote_final, laplace = 0, usekernel = FALSE, adjust = 1,weights = weights)

#Score!
nb_predict <- predict(NB_Model, type= 'prob')[,2]
plotROC(smote_final$Class, nb_predict) #97.25%
nb_eval <- evalmod(scores = nb_predict, labels = smote_final$Class)
auc(nb_eval) #87.59003%
ks_stat(actuals =smote_final$Class, predictedScores = nb_predict) #0.9554
nb_predict <- predict(NB_Model, type= 'class')
caret::confusionMatrix(nb_predict, smote_final$Class)
#Specificity:.9863
#Sensitivity: .9605

#########################Results from Training!##########################

##LR##
# AUROC 98.53%
# AUPRC 99.42303%
# KS: 0.9528
#Specificity: 0.9998
#Sensitivity: 0.9250

##Neural Network##
# AUROC: 99.89%
# AUPRC: 99.71933%
# KS: 0.9601
# Specificity: 0.9996
# Sensitivity: 0.9957

##Random Forest##
# AUROC: 99.84%
# AUPRC: 99.72120%
# KS: 0.9578
# Specificity:.9999
# Sensitivity:.9947

##XGBoost##
# AUROC: 100%
# AUPRC: 100%
# KS: 0.9609
# Specificity: 1 
# Sensitivity: 1

##Naive Bayes##
# AUROC: 97.25%
# AUPRC: 87.59003%
# KS: 0.9554
# Specificity:.9863
# Sensitivity: .9605

#Hard to argue against XGBoost being the best, perfectly classified

###################### Test XGBoost on Test Data ######################
#remove test data set index and make class a factor
test <- test[-1]
test$Class <- as.factor(test$Class)
testx <- model.matrix(Class~.,data=test)[,-31]
#get prediction object
test$p_hat <- predict(XGB_Model, newdata = testx, type= 'raw')

#Coefficient of Discrimination
p1 <- test$p_hat[test$Class == 1]
p0 <- test$p_hat[test$Class == 0]
mean(p1) - mean(p0) #.8330694

#Make ROC
plotROC(test$Class, test$p_hat) #93.53%
final_eval <- evalmod(scores = test$p_hat, labels = test$Class)
auc(final_eval) #87.85655%
ks_stat(actuals =test$Class, predictedScores = test$p_hat) #.8413
#basic threshold of .5
xgb_test <- as.numeric(test$p_hat > 0.5)
caret::confusionMatrix(as.factor(xgb_test), test$Class)
#specificity: 0.9999
#Sensitivity: 0.8448
#Classified 98 of 116 frauds correctly
#Classified 59386 of 59393 not frauds correct

#lets use "optimal" threshold that maximizes TPR and FPR difference

pred <- prediction(test$p_hat, test$Class)
perf <- performance(pred, measure = 'tpr', x.measure = 'fpr')
unlist(perf@alpha.values)[which.max(perf@y.values[[1]] - perf@x.values[[1]])]
#0.000512919 threshold identified
xgb_test <- as.numeric(test$p_hat > 0.000512919)
caret::confusionMatrix(as.factor(xgb_test), test$Class)
#Specificity: 0.9886
#Sensitivity: 0.9138

#Lets try to find threshold that makes the most sense to the problem
xgb_test <- as.numeric(test$p_hat > 0.00255)
caret::confusionMatrix(as.factor(xgb_test), test$Class)
#Specificity: 0.9978
#Sensitivity: 0.9052

#Classified 105 out of 116 frauds correctly
#Classified 59263 of 59393 not frauds correct

#Fin


