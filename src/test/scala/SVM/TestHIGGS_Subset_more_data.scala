package SVM

import SVM.DataSetType.{Test, Validation}

import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}

object TestHIGGS_Subset_more_data extends App {

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/HIGGS/"
  val pathTrain = workingDir + "higgsTrain2.csv"
  val pathValidation = workingDir + "higgsValidation2.csv"
  val pathTest = workingDir + "higgsTest2.csv"

  //I have to define a transform function because the label codes do not correspond to the default (+1 for signal and -1 for noise)
  val transformLabel = (x:Double) => if(x<=0) -1 else +1
  //the labels are in the second column (the column index is 0 based)
  val columnIndexLabel = 1
  val columnIndexLineNr = 0
  //The first column has to be skipped, it contains a line nr!!!
  d.readTrainingDataSet (pathTrain, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readTestDataSet (pathTest, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.readValidationDataSet(pathValidation, ',', columnIndexLabel, transformLabel, columnIndexLineNr)
  d.tableLabels()

  val medianScale = d.probeKernelScale()
  println("The kernel scale parameter was estimated at "+medianScale+ " from the training data.")

  d.selectInstances(sampleProb=0.1, minQuantile=0.45, maxQuantile=0.55)

  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(medianScale)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)

  val mp = ModelParams(C = 1.0, delta = 0.01)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(batchProb = 0.99, learningRateDecline = 0.8,
    epsilon = epsilon)
  var algo = NoMatrices(alphas, ap, mp, kmf, new ListBuffer[Future[(Int,Int,Int)]])
  var numInt = 0
  while(numInt < ap.maxIter){
    algo = algo.iterate(numInt)
    numInt += 1
  }

  val testSetAccuracy : Future[Int] = algo.predictOn(Validation, PredictionMethod.AUC)
  Await.result(testSetAccuracy, LeanMatrixFactory.maxDuration)

  val testSetAccuracy2 : Future[Int] = algo.predictOn(Test, PredictionMethod.THRESHOLD,0.60)
  Await.result(testSetAccuracy2, LeanMatrixFactory.maxDuration)
}