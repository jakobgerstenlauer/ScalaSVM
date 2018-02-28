library(MASS)
library(kernlab)
library(caret)
set.seed(5678)
setwd("~/workspace_scala/Dist_Online_SVM/R")
source("paths.R")
setwd(higgsDir)
dtrain<-read.table("higgsTrain.csv",sep=",",header=FALSE)
dim(dtrain)
#[1] 40000   30

#The first column contains only the line nr and can be omitted:
#For a test run:
#dtrain <- dtrain[1:500,-1]
dtrain <- dtrain[,-1]
#The second column (now the first) contains the class labels as 0 and 1.
#The following 28 columns are the numeric inputs.
names(dtrain)<-c("class",glue("input",1:28))

start.time <- Sys.time()
svm <- ksvm(as.factor(class)~., data=dtrain, kernel='rbfdot',  C=100, scaled=c(), type="C-svc") 
end.time <- Sys.time()
print(end.time - start.time)
#Time difference of 1.771862 hours

#Read the test data:
dtest <- read.table("higgsTest.csv",sep=",",header=FALSE)
dim(dtest)
#The first column contains only the line nr and can be omitted:
dtest <- dtest[,-1]
names(dtest)<-c("class",glue("input",1:28))
preds <- predict(svm, dtest)
(confusionMatrix <- table(preds, dtest$class))
# preds     0     1
# 0 11355  6257
# 1  7391 14997
(accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix))
# [1] 0.6588
 svm
# Support Vector Machine object of class "ksvm" 
# 
# SV type: C-svc  (classification) 
# parameter : cost C = 100 
# 
# Gaussian Radial Basis kernel function. 
# Hyperparameter : sigma =  0.0298888743702596 
# 
# Number of Support Vectors : 27781 
# 
# Objective Function Value : -1748409 
# Training error : 0.134325 

distClass<-table(dtest$class)
#false positive rate:
7391/21254
