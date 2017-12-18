package SVM

object TestLocalAlgorithm extends App {

  val dataProperties = DataParams(N=19020, d=10, ratioTrain=0.5)
  println(dataProperties)

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathTest = workingDir + "magic04test.csv"

  d.readTrainingDataSet (pathTrain, ',', 10)
  d.readTestDataSet (pathTest, ',', 10)
  println(d)
  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(4.0)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = new LocalKernelMatrixFactory(d, gaussianKernel, epsilon)
  val alphas = new Alphas(N=d.N_train)
  val ap = AlgoParams(maxIter = 5, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.9,
    numBaggingReplicates = 50, epsilon = epsilon, isDebug = true, hasMomentum = false)
  val mp = ModelParams(C = 1.0, lambda = 10.0, delta = 0.3)
  var algo1 = new SGLocal(alphas, ap, mp, kmf)
  var numInt = 0
  while(numInt < 4){
    algo1 = algo1.iterate()
    numInt += 1
    println(numInt)
  }
}