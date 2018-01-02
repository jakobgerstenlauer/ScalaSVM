package test
import SVM._
import SVM.DataSetType.{Test, Train}

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
	val dataProperties = DataParams(N = N, d = 5, ratioTrain = 0.5)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)

  //First find a value for epsilon that is manageable:
	val probeMatrices = ProbeMatrices(d, gaussianKernel)

	//Number of non-sparse matrix elements with epsilon = 0.001:
	val epsilon = 0.001
	val numElementsS =  probeMatrices.probeSparsity(Test, 0.001)
	val numElementsK =  probeMatrices.probeSparsity(Train, 0.001)
  println("Projected memmory requirements for epsilon ="+epsilon+":")
  //Integer = 32 bits = 4 Byte
  val intsPerKB = 256
  println("Training matrix K: "+numElementsK/intsPerKB+"kB:")
  println("Training matrix S: "+numElementsS/intsPerKB+"kB:")

  val lmf = time{LeanMatrixFactory(d, gaussianKernel, epsilon)}
	val alphas = new Alphas(N=N/2)
	val ap = AlgoParams(maxIter = 30, batchProb = 0.6, minDeltaAlpha = 0.001, learningRateDecline = 0.7, epsilon = epsilon, isDebug = false, hasMomentum = false, sparsity=0.1)
	val mp = ModelParams(C = 0.2, delta = 0.1)
	var algo = new NoMatrices(alphas, ap, mp, lmf)
	var numInt = 0
	while(alphas.getDelta > 0.01 && numInt < ap.maxIter){
		algo = time{algo.iterate()}
		numInt += 1
	}
}
