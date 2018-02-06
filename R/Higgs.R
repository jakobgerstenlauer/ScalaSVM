rm(list=ls(all=TRUE))
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
dtrain <- dtrain[,-1]
#The second column (now the first) contains the class labels as 0 and 1.
#The following 28 columns are the numeric inputs.
names(dtrain)<-c("class",glue("input",1:28))

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
(bestcv <- which(cverrors == min(cverrors), arr.ind = T))
#The optimal C is 100.
