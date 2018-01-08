package SVM

object TestLocalAlgorithm extends App {

  val dataProperties = DataParams(N=19020, d=10, ratioTrain=0.5)
  println(dataProperties)

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathTest = workingDir + "magic04test.csv"

  d.readTrainingDataSet (pathTrain, ',', 11)
  d.readTestDataSet (pathTest, ',', 11)
  println(d)
  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(10.0)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 0.1, delta = 0.3)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter = 5, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.9,
    epsilon = epsilon, isDebug = true, hasMomentum = false)
  var algo1 = NoMatrices(alphas, ap, mp, kmf)
  var numInt = 0
  while(numInt < 2){
    algo1 = algo1.iterate()
    numInt += 1
    println(numInt)
  }
}
