# ScalaSVM
This is a Scala implementation of kernelized support vector machines for binary classification based on a stochastic gradient descent algorithm.

## Background Information
Kernelized machine learning methods are able to predict highly nonlinear relationships for both numeric input data and heterogeneous symbolic data types such as strings, protein and DNA sequences, images, and graphs. Established algorithms such as interior point algorithms and sequential minimal optimization are very accurate and fast, but they do not scale to big data situations where data processing has to take place in a distributed system. In this Scala project, a stochastic gradient descent algorithm was developed that solves the kernelized support vector machine in the dual formulation. This core algorithm can be used in a local parallelized and two different distributed implementations. The parallelized version implements a lean sparse matrix representation using hash maps and is able to concurrently calculate partial gradients. The two distributed versions use distributed matrices provided by Apache Spark to scale out the evaluation of the gradient and the evaluation of the model coefficients on the training and validation set. 

Apart from the algorithm that learns the support vector machine, additional tools are provided: First, a subsection selection heuristic that uses a efficient projection method to filter instances. Second, a set of representative kernels and a heuristic for finding an appropriate value for the Gaussian kernel. Third, a visualization tool that helps to evaluate the quality of the classifier and decide on a decision threshold based on a receiver operator characteristic (ROC) curve for the validation set. And finally, a general cross-validation scheme that i) implements an early stopping strategy by selecting the model coefficients from the optimal iteration, ii) finds the optimal level of sparsity, iii) and helps to decide on a useful decision threshold. Asynchronous programming principles are used, to accelerate the computations and facilitate parallelization. 

## Usage
### Creating artifical data sets
In order to try out the library, it is possible to create artifical data sets with a given number of observations *N* and features *d* and a binary output: 
```scala
import SVM._
val N = 200000
val dataProperties = DataParams(N = N, d = 10)
val d = new SimData(dataProperties)
d.simulate()
```    

### Reading in empirical data sets
### Defining the kernel function
### 
