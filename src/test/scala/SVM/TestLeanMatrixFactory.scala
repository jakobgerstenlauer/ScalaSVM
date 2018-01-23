package test
import org.scalatest.FunSuite
import SVM._

import scala.collection.mutable.ListBuffer
import scala.concurrent.Future

class TestLeanMatrixFactory extends FunSuite{

  test("Algo must run locally with simulated data."){
	val kernelPar = GaussianKernelParameter(1.0)
	val gaussianKernel = GaussianKernel(kernelPar)
	val N = 10
	val d = new SimData(DataParams(N = N, d = 10))
	d.simulate()
	val epsilon = 0.001
        val lmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
	val mp = ModelParams(C = 0.5, delta = 0.1)
	val alphas = new Alphas(N=N/2, mp)
	val ap = AlgoParams(maxIter = 20, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.5, epsilon = epsilon, quantileAlphaClipping=0.03)
	var algo = NoMatrices(alphas, ap, mp, lmf, new ListBuffer[Future[(Int,Int,Int)]])
	var numInt = 0
  	while(numInt < ap.maxIter && algo.getSparsity < 99.0){
		algo = algo.iterate(numInt)
		numInt += 1
	}
  }
}
