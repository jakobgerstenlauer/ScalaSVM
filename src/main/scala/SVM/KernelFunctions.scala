package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps
import breeze.linalg.operators

abstract class KernelParameter{
}

case class GaussianKernelParameter(sigma: Double) extends KernelParameter{
	override def toString() : String = {
        	return "Gaussian kernel parameter sigma "+sigma+". \n"
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
  		 assert(kernelParam.sigma>0.0, "Sigma must be positive!")
  		 val diff = x - y
		 val squares = diff :* diff
  		 val squared_euclidean_distance = sum(squares)
  		 return exp( -kernelParam.sigma * squared_euclidean_distance)
	}
	override def toString() : String = {
        	val sb = new StringBuilder
        	sb.append("Gaussian kernel: \n")
	        sb.append("sigma: "+kernelParam.sigma+"\n")
	        return sb.toString()
  	}  
}

