rm(list=ls(all=TRUE))
library(MASS)
library(kernlab)
library(caret)
set.seed(5678)
setwd("~/workspace_scala/Dist_Online_SVM/R")
source("paths.R")
setwd(magicDir)
dtrain<-read.table("magic04train.csv",sep=",",header=FALSE)
dim(dtrain)
#[1] 9508   12

#The first column contains only the line nr and can be omitted:
dtrain <- dtrain[,-1]
names(dtrain)<-c(glue("input",1:10),"class")

# Generating the same CV folds for every method
folds <- createFolds(dtrain$class, k = 10)

# Function to try several parameters for the SVM with CV
# Original author: Martí Zamora
svm.tune <- function(formula, data, kernels, Cs, folds) {
  res <- matrix(nrow=length(kernels), ncol=length(Cs))
  n_folds = length(folds)
  for (i in 1:length(kernels)){
    k = kernels[[i]]
    for(j in 1:length(Cs)){
      C = Cs[j]
      print(paste(i,j,k$kernel))
      cverror <- rep(NA, n_folds)
      for(f in 1:n_folds){
        if(is.null(k$kpar)){
          svm <- ksvm(formula, data=data[-folds[[f]],], kernel=k$kernel,  C=C, scaled=c(), type="C-svc") 
        }else{
          svm <- ksvm(formula, data=data[-folds[[f]],], kernel=k$kernel, kpar=k$kpar,  C=C, scaled=c()) 
        }
        preds <- predict(svm, data[folds[[f]],])
        confusionMatrix <- table(preds, data[folds[[f]],]$class)   
        cverror[f] <- 1 - sum(diag(prop.table(confusionMatrix)))
      }
      res[i,j] <- mean(cverror)
    }
  }
  res 
}

# List of kernels to try
SVMKernels = list(
  list(kernel='rbfdot')
)

# Trying C with values from 0.01 to 100
CsExponents <- c(-2:2)
Cs <- 10**CsExponents

(cverrors <- svm.tune(as.factor(class)~., data=dtrain, kernels=SVMKernels, Cs=Cs, folds=folds))
# [,1]      [,2]      [,3]     [,4]      [,5]
# [1,] 0.2131892 0.1811113 0.1687016 0.162181 0.1593411

(bestcv <- which(cverrors == min(cverrors), arr.ind = T))
# row col
# [1,]   1   5
#The optimal C is 100.

#****************************************************************************************************
#Let's rerun with a smaller range of possible Cs and the combined training and validation data set:
Cs<-c(60,80,100,120)

dtrain<-read.table("magic04train.csv",sep=",",header=FALSE)
dim(dtrain)
#[1] 9508   12

dval<-read.table("magic04validation.csv",sep=",",header=FALSE)
dim(dval)

dtrain<-rbind(dtrain,dval)

#The first column contains only the line nr and can be omitted:
dtrain <- dtrain[,-1]
names(dtrain)<-c(glue("input",1:10),"class")

# Generating the same CV folds for every method
folds <- createFolds(dtrain$class, k = 10)

(cverrors <- svm.tune(as.factor(class) ~ . , data=dtrain, kernels=SVMKernels, Cs=Cs, folds=folds))
# [,1]      [,2]      [,3]      [,4]
# [1,] 0.1594914 0.1586502 0.1561263 0.1568276
(bestcv <- which(cverrors == min(cverrors), arr.ind = T))
# row col
# [1,]   1   3
svm <- ksvm(as.factor(class)~., data=dtrain, kernel='rbfdot',  C=100, scaled=c(), type="C-svc") 

#Read the test data:
dtest <- read.table("magic04test.csv",sep=",",header=FALSE)
dim(dtest)
#The first column contains only the line nr and can be omitted:
dtest <- dtest[,-1]
names(dtest)<-c(glue("input",1:10),"class")
preds <- predict(svm, dtest)
(confusionMatrix <- table(preds, dtest$class))
# preds   -1    1
# -1 1095  173
# 1   579 2909
(accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix))
#[1] 0.8418839
svm
# Support Vector Machine object of class "ksvm" 
# 
# SV type: C-svc  (classification) 
# parameter : cost C = 100 
# 
# Gaussian Radial Basis kernel function. 
# Hyperparameter : sigma =  0.000158682277197226 
# 
# Number of Support Vectors : 5119 
# 
# Objective Function Value : -413710.1 
# Training error : 0.121775 