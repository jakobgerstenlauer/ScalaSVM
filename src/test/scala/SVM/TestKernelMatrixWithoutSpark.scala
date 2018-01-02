package test
import SVM._

object testKernelMatrixWithoutSpark extends App {
  //http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }
	val kernelPar = GaussianKernelParameter(1.0)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
	val N = 10000
	val dataProperties = DataParams(N = N, d = 10, ratioTrain = 0.5)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)
	val epsilon = 0.001
	val lmf = time{LeanMatrixFactory(d, gaussianKernel, epsilon)}
	val alphas = new Alphas(N=N/2)
	val ap = AlgoParams(maxIter = 10, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.5, epsilon = epsilon, isDebug = false, hasMomentum = false, sparsity=0.0001)
	val mp = ModelParams(C = 0.2, delta = 0.1)
	var algo = new NoMatrices(alphas, ap, mp, lmf)
	var numInt = 0
	while(alphas.getDelta > 0.01 && numInt < ap.maxIter){
		algo = time{algo.iterate()}
		numInt += 1
	}
}
