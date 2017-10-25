package test
import SVM.KernelParameter
import SVM.GaussianKernelParameter
import SVM.KernelFunction
import SVM.GaussianKernel
import breeze.linalg._
import breeze.numerics._

object testKernel extends App {
	val kernelPar = GaussianKernelParameter(1.5)
	val gaussianKernel = GaussianKernel(kernelPar)
	val length = 10
	val x = DenseVector.rand(length)
	val y = DenseVector.rand(length)
	println("x:"+x)
	println("y:"+y)
	println("k(x,y)="+gaussianKernel.kernel(x,y))
}
