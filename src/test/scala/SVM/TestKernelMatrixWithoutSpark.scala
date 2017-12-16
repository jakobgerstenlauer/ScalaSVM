package test
import SVM.{DataParams, GaussianKernel, GaussianKernelParameter, SimData}

object testKernelMatrixWithoutSpark extends App {
	val kernelPar = GaussianKernelParameter(1.5)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
 	val dataProperties = DataParams(N=100, d=10, ratioTrain=0.5)
	println(dataProperties)
        val d = new SimData(dataProperties)
	println(d)
        d.simulate()
	println(d)
}
