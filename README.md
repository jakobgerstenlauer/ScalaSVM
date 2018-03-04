# ScalaSVM
This is a Scala implementation of kernelized support vector machines for binary classification based on a stochastic gradient descent algorithm.

## Background Information
Kernelized machine learning methods are able to predict highly nonlinear relationships for both numeric input data and heterogeneous symbolic data types such as strings, protein and DNA sequences, images, and graphs. Established algorithms such as interior point algorithms and sequential minimal optimization are very accurate and fast, but they do not scale to big data situations where data processing has to take place in a distributed system. In this Scala project, a stochastic gradient descent algorithm was developed that solves the kernelized support vector machine in the dual formulation. This core algorithm can be used in a local parallelized and two different distributed implementations. The parallelized version implements a lean sparse matrix representation using hash maps and is able to concurrently calculate partial gradients. The two distributed versions use distributed matrices provided by Apache Spark to scale out the evaluation of the gradient and the evaluation of the model coefficients on the training and validation set. 

Apart from the algorithm that learns the support vector machine, additional tools are provided: First, a subsection selection heuristic that uses a efficient projection method to filter instances. Second, a set of representative kernels and a heuristic for finding an appropriate value for the Gaussian kernel. Third, a visualization tool that helps to evaluate the quality of the classifier and decide on a decision threshold based on a receiver operator characteristic (ROC) curve for the validation set. And finally, a general cross-validation scheme that i) implements an early stopping strategy by selecting the model coefficients from the optimal iteration, ii) finds the optimal level of sparsity, iii) and helps to decide on a useful decision threshold. Asynchronous programming principles are used, to accelerate the computations and facilitate parallelization. 

## Usage
### Creating artifical data sets
In order to get started with the library, it is possible to create artifical data sets with a binary output (label) and a given number of observations *N* and features *d*: 
```scala
import SVM._
val N = 200000
val dataProperties = DataParams(N = N, d = 10)
val d = new SimData(dataProperties)
d.simulate()
```    
By default, 50% of the instances are assigned to a training set, 10% are assigned to a test set, and 40% are assigned to a validation set. The user can specify different ratios using the *ratioTrain* and *ratioTest* arguments of the *DataParams* constructor:
```scala
val dataProperties = DataParams(N = N, d = 10, ratioTrain = 0.8, ratioTest: Double = 0.1)
```    

### Reading in empirical data sets
When fitting a support vector machine locally on a data set, it is assumed that the user can separate the data set into three separate csv text files for the training, validation and test set. The input files should not have a header and consist only of numeric data types. Any separator can be specified via the *separator* argument. The user additionally has to specify the complete path to the input files as string and the index of class labels and any column that should be ignored using zero-based indexing. It is implicitly assumed that the class labels are +1 for the signal and -1 for the background class. If the class labels follow a different code, a anonymous function has to be specified which transforms the labels into the correct code (in the example code this function is called *transLabel*).

```scala
  val workingDir = "/home/user/data/"
  val pathTrain = workingDir + "magicTrain.csv"
  val pathValidation = workingDir + "magicValidation.csv"
  val pathTest = workingDir + "magicTest.csv"
  val indexLabel = 11 
  val indexSkip = 0 //The first column has to be skipped (line nr!)
  val transLabel = (x:Double) => if(x<=0) -1 else +1
  val d = new LocalData()
  d.readTrainingDataSet (pathTrain, ',', indexLabel, transLabel, indexSkip)
  d.readTestDataSet (pathTest, ',', indexLabel, transLabel, indexSkip)
  d.readValidationDataSet(pathValidation, ',', indexLabel, transLabel, indexSkip)
```    
After having read the input files, the user may want to print the class distribution for all data sets:
```scala
  d.tableLabels()
```    

### Defining the kernel function 
The function *probeKernelScale()* can be used to determine a useful estimate for the kernel parameter of the Gaussian kernel function:
```scala
  val medianScale = d.probeKernelScale()
  println("Estimated kernel scale parameter: "+medianScale)
```    
Now, a Gaussian kernel function can be defined with an appropriate sparsity threshold epsilon:
```scala
  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(medianScale)
  val gaussianKernel = GaussianKernel(kernelPar)
```    
Based on the Gaussian kernel, a local representations of the kernel matrices for the three data sets is created:
```scala
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
```    
Apart from the Gaussian kernel, the linear and the polynomial kernel are also available and they can be used equivalently:

```scala
  val epsilon = 0.0001
  val scale = 0.0
  val kernelParPoly = PolynomialKernelParameters(scale=1.0, offset=0.0, degree=3.0)
  val polynomialKernel = PolynomialKernel(kernelParPoly)
  val kmf = LeanMatrixFactory(d, polynomialKernel, epsilon)  
```    

### Running a local SVM algorithm

The algorithm itself is initiated given this matrix factory object, a new *Alphas* object and a *ModelParams* object which bundles the parameters *C* and the learning rate delta. 
```scala
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter=10,batchProb = 0.99,learningRateDecline = 0.8,epsilon = epsilon)
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
```    
The algorithm is iterated using a loop construct. At the end of the loop, a blocking *Await.result()* is needed to keep the main thread from shutting down before the parallel evaluation of the final model on the test set has finalized:
```scala
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
  var numInt = 0
  while(numInt < ap.maxIter){
    algo = algo.iterate(numInt); numInt += 1
  }
  Await.result(algo.predictOn(Test, PredictionMethod.STANDARD), LeanMatrixFactory.maxDuration)
```    

The complete code example as a self-contained local Scala application is:
```scala
package SVM
import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}
import SVM.DataSetType.{Test, Train, Validation}

object TestMAGIC extends App {
  val workingDir = "/home/user/data/"
  val pathTrain = workingDir + "magicTrain.csv"
  val pathValidation = workingDir + "magicValidation.csv"
  val pathTest = workingDir + "magicTest.csv"
  val indexLabel = 11 
  val indexSkip = 0 //The first column has to be skipped (line nr!)
  val transLabel = (x:Double) => if(x<=0) -1 else +1
  val d = new LocalData()
  d.readTrainingDataSet (pathTrain, ',', indexLabel, transLabel, indexSkip)
  d.readTestDataSet (pathTest, ',', indexLabel, transLabel, indexSkip)
  d.readValidationDataSet(pathValidation, ',', indexLabel, transLabel, indexSkip)
  d.tableLabels()
  val medianScale = d.probeKernelScale()
  println("Estimated kernel scale parameter: "+medianScale)
  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(medianScale)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 100, delta = 0.01)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter=10,batchProb = 0.99,learningRateDecline = 0.8,epsilon = epsilon)
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
  var numInt = 0
  while(numInt < ap.maxIter){
    algo = algo.iterate(numInt); numInt += 1
  }
  Await.result(algo.predictOn(Test, PredictionMethod.STANDARD), LeanMatrixFactory.maxDuration)
}
```    
### Running a distributed SVM algorithm

In order to use the distributed algorithms, the user needs access to a local Spark cluster or a commercial Spark cloud solution. Depending on the number of central processing units (CPUs) available to the driver node, it is recommended to use the sequential *SG* or the parallelized *SGwithFutures* algorithm. Some code snippets for the sequential algorithm are shown below. 

First, an artificial data set is created:
```scala
import SVM._
val N = 40000
val kernelPar = GaussianKernelParameter(1.0)
val gaussianKernel = GaussianKernel(kernelPar)
val ratioTrainingObservations=0.5
val dataProperties = DataParams(N=N, d=10, ratioTrain=ratioTrainingObservations)
val d = new SimData(dataProperties)
d.simulate()
```    
Then, the distributed matrices for the training set, validation set, and test set are created:
```scala
val epsilon = 0.001
//returns the underlying SparkContext
val sc = spark.sparkContext
val kmf = new KernelMatrixFactory(d, gaussianKernel, epsilon, sc)
```    
Now, the model itself and the algorithm is created:
```scala
import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}
val mp = ModelParams(C = 0.5, delta = 0.01)
val alphas = new Alphas(N=N/2, mp)
val ap = AlgoParams(maxIter = 30, batchProb = 0.99, learningRateDecline = 0.8, epsilon = epsilon, quantileAlphaClipping=0.0)
var algo1 = new SG(alphas, ap, mp, kmf, sc, new ListBuffer[(Int,Int)])
```    
By default, the distributed algorithm does not enforce sparsity. However, sparsity can be enforced using the argument *quantileAlphaClipping*. Setting this parameter to 0.1 enforces a sparsity of 10%:
```scala
val ap = AlgoParams(maxIter = 30, batchProb = 0.99, learningRateDecline = 0.8, epsilon = epsilon, quantileAlphaClipping=0.1)
```   
The iterative gradient descent optimization is run with:
```scala
var numIt = 0
while(numIt < 5){
  algo1 = algo1.iterate(numIt)
  numIt += 1
}
```    
### Deciding on a classification threshold
 
After training the support vector machine on the training set, it is possible to evaluate the final model on the validation set for all percentiles of the dual variables:
```scala
val future = algo.predictOn(Validation, PredictionMethod.AUC)
	Await.result(future, LeanMatrixFactory.maxDuration)
```
The function then prints both a graphical and a text representation of the ROC curve which enables the user to decide on the optimal decision threshold.


