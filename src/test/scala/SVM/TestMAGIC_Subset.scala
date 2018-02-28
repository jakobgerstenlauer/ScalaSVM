package SVM

import SVM.DataSetType.{Test, Validation}

import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}

object TestMAGIC_Subset extends App {

  val d = new LocalData()
  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathValidation = workingDir + "magic04validation.csv"
  val pathTest = workingDir + "magic04test.csv"

  //The labels are in the second column (the column index is 0 based)
  val indexLabel = 11
  //The first column has to be skipped, it contains a line nr!!!
  val indexSkip = 0
  val transLabel = (x:Double) => if(x<=0) -1 else +1
  d.readTrainingDataSet(pathTrain, ',', indexLabel, transLabel, indexSkip)
  d.readTestDataSet (pathTest, ',', indexLabel, transLabel, indexSkip)
  d.readValidationDataSet(pathValidation, ',', indexLabel, transLabel, indexSkip)
  d.tableLabels()

  val medianScale = d.probeKernelScale()

  println("The kernel scale parameter was estimated at "+medianScale+ " from the training data.")
  //println("The lower 0.1% quantile of the Euclidean distance of a subset of training set instances was used as estimate for the Epsilon sparsity threshold parameter: " + epsilon)

  d.selectInstances()

  val epsilon = 0.001
  val kernelPar = GaussianKernelParameter(medianScale)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 100, delta = 0.01)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter=10, batchProb = 0.99, learningRateDecline = 0.8,
    epsilon = epsilon)
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
  var numInt = 0
  while(numInt < ap.maxIter){
    algo = algo.iterate(numInt)
    numInt += 1
  }

  val testSetAccuracy : Future[Int] = algo.predictOn(Validation, PredictionMethod.AUC)
  Await.result(testSetAccuracy, LeanMatrixFactory.maxDuration)

  val testSetAccuracy2 : Future[Int] = algo.predictOn(Test, PredictionMethod.THRESHOLD,0.67)
  Await.result(testSetAccuracy2, LeanMatrixFactory.maxDuration)
}