import SVM.GaussianKernelParameter
import SVM.GaussianKernel
import breeze.linalg._

object TestGaussianKernel extends App {
	val kernelPar = GaussianKernelParameter(1.5)
	val gaussianKernel = GaussianKernel(kernelPar)
	val length = 10
	val x : DenseVector[Double]= DenseVector.rand(length)
	val y : DenseVector[Double]= DenseVector.rand(length)
	println("x:"+x)
	println("y:"+y)
  println("k(x,y)="+gaussianKernel.kernel(x,y))
}
