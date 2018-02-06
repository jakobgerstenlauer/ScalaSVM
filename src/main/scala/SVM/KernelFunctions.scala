package SVM

import breeze.linalg._
import breeze.numerics._

abstract class KernelParameter{
}

case class GaussianKernelParameter(gamma: Double) extends KernelParameter{
	override def toString : String = "Gaussian kernel parameter sigma "+gamma+". \n"
}

case class PolynomialKernelParameters(scale: Double, offset: Double, degree: Double) extends KernelParameter{
	override def toString : String = "Polynomial kernel parameter with scale: "+scale+", offset: "+offset+", degree: "+degree+". \n"
}

abstract class KernelFunction{
	//val kernelParam : KernelParameter
	//Calculates the kernel function for two observations
	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double
}

case class GaussianKernel(kernelParam : GaussianKernelParameter) extends KernelFunction{

	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double ={
		assert(x.length == y.length, "Incompatible vectors x and y in rbf() function!")
		assert(kernelParam.gamma>0.0, "Sigma must be positive!")
		val diff = x - y
		val squares = diff *:* diff
		val squared_euclidean_distance = sum(squares)
		exp( -kernelParam.gamma * squared_euclidean_distance)
	}

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Gaussian kernel: \n")
		sb.append("Gamma: "+kernelParam.gamma+"\n")
		sb.toString()
	}
}

case class PolynomialKernel(kernelParam : PolynomialKernelParameters) extends KernelFunction{

	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double ={
		assert(x.length == y.length, "Incompatible vectors x and y in kernel() function!")
		val base = kernelParam.offset + kernelParam.scale * (x dot y)
		Math.pow(base, kernelParam.degree)
	}

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Polynomial kernel: \n")
		sb.append(kernelParam.toString())
		sb.toString()
	}
}

case class LinearKernel() extends KernelFunction{

	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double ={
		assert(x.length == y.length, "Incompatible vectors x and y in rbf() function!")
		x dot y
	}

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Linear kernel (no parameters). \n")
		sb.toString()
	}
}
