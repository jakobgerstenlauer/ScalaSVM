package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps
import breeze.linalg.operators

abstract class KernelParameter{
	def showParameters() : Unit
}
case class GaussianKernelParameter(gamma: Double) extends KernelParameter{
	def showParameters() : Unit = {
		println("The kernel parameter gamma is: "+gamma)
	}
}

abstract class KernelFunction{
	val kernelParam : KernelParameter
	//Calculates the kernel function for two observations
	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double
}

case class GaussianKernel(kernelParam : GaussianKernelParameter) extends KernelFunction{
	def kernel(x: DenseVector[Double], y: DenseVector[Double]) : Double ={
		 assert(x.length == y.length, "Incompatible vectors x and y in rbf() function!")
  		 assert(kernelParam.gamma>0.0, "Gamma must be positive!")
  		 val diff = x - y
		 val squares = diff :* diff
  		 val squared_euclidean_distance = sum(squares)
  		 return exp( -kernelParam.gamma * squared_euclidean_distance)
	}
}

