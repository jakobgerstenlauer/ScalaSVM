library(MASS)
library(kernlab)
library(caret)
set.seed(5678)
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
# [,1]      [,2]      [,3]      [,4]      [,5]
# [1,] 0.9991587 0.9990534 0.9990534 0.9988431 0.9989483

(bestcv <- which(cverrors == min(cverrors), arr.ind = T))
# row col
# [1,]   1   4

Cs
#[1] 1e-02 1e-01 1e+00 1e+01 1e+02
#The optimal C is 10.


#****************************************************************************************************
#Let's rerun with a smaller range of possible Cs and the combined training and validation data set:
Cs<-c(4,8,10,12)

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
# [1,] 0.9992989 0.9992989 0.9992989 0.9992989
(bestcv <- which(cverrors == min(cverrors), arr.ind = T))
# row col
# [1,]   1   1
# [2,]   1   2
# [3,]   1   3
# [4,]   1   4

#There is no difference between the values of C!
svm <- ksvm(as.factor(class)~., data=dtrain, kernel='rbfdot',  C=10, scaled=c(), type="C-svc") 
# Support Vector Machine object of class "ksvm" 
# 
# SV type: C-svc  (classification) 
# parameter : cost C = 10 
# 
# Gaussian Radial Basis kernel function. 
# Hyperparameter : sigma =  0.000151297039339112 
# 
# Number of Support Vectors : 5421 
# 
# Objective Function Value : -47843.84 
# Training error : 0.14477 

#Read the test data:
dtest <- read.table("magic04test.csv",sep=",",header=FALSE)
dim(dtest)
#The first column contains only the line nr and can be omitted:
dtest <- dtest[,-1]
names(dtest)<-c(glue("input",1:10),"class")
preds <- predict(svm, dtest)
(confusionMatrix <- table(preds, dtest$class))
# preds   -1    1
# -1 1047  154
# 1   627 2928
