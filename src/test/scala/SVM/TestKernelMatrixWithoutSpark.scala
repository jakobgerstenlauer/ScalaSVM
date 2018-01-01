package test
import SVM._

/**
	* Here I compare the computation time using the linear loop approach of initializeRowColumnPairs()
	* with the parallelized streaming approach of initializeRowColumnPairs2().
	* I also check for identical results (identical Map).
	*/
object testKernelMatrixWithoutSpark extends App {
	val kernelPar = GaussianKernelParameter(2.0)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
	val dataProperties = DataParams(N = 50000, d = 10, ratioTrain = 0.5)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)
	val epsilon = 0.001
	val lmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
	//println(lmf.rowColumnPairs==lmf.rowColumnPairs2)
	//println(lmf.rowColumnPairs.equals(lmf.rowColumnPairs2))
}
