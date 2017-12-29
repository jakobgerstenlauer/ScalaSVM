package test
import SVM._
import breeze.linalg._
import breeze.numerics._

object TestAlgoWithoutSpark extends App {
  val N = 1000
  val kernelPar = GaussianKernelParameter(1.5)
  val gaussianKernel = GaussianKernel(kernelPar)
  val dataProperties = DataParams(N=N, d=10, ratioTrain=0.5)
  val d = new SimData(dataProperties)
  d.simulate()
  val epsilon = 0.00001
  val lkmf = LocalKernelMatrixFactory(d, gaussianKernel, epsilon)
  val alphas = new Alphas(N=(0.5*N).toInt)
  val ap = AlgoParams(maxIter = 30, minDeltaAlpha = 0.001, learningRateDecline = 0.9,
    numBaggingReplicates = 30, batchProb = 0.5, epsilon = epsilon, isDebug = true, hasMomentum = false)
  val mp = ModelParams(C = 10.0, lambda = 100.0, delta = 0.1)
  var algo1 = new SGLocal(alphas, ap, mp, lkmf)
  var numInt = 0
  while(numInt < ap.maxIter){
      algo1 = algo1.iterate()
      numInt += 1
  }
}
