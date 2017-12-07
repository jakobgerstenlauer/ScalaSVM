package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps 
import breeze.linalg.operators 
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import collection.mutable.{HashMap, MultiMap, Set}

case class LocalKernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double){

def calculateGradient(alphas : DenseVector[Double]) : DenseVector[Double]  = {
  val N = d.getN_train()
  var v = DenseVector.zeros[Double](N)
  for (i <- 0 until N; j <- 0 until N){
    v(i) += alphas(j) * d.z_train(i) * d.z_train(j) * kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t)
  }
  v - DenseVector.ones[Double](N)
}

}

case class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext){

val matOps = new DistributedMatrixOps(sc)
val K = initKernelMatrixTraining()
val S = initKernelMatrixTest()
val Z = initTargetMatrixTraining()
val Z_test = initTargetMatrixTest()
val z = initTargetTraining()
val z_test = initTargetTest()

def getData() : Data = return d

private def initTargetMatrixTraining() : CoordinateMatrix = {
  assert( d.isValid() , "The input data is not defined!")
  matOps.distributeTranspose(d.z_train)
}
         
private def initTargetMatrixTest() : CoordinateMatrix = {
  assert( d.isValid() , "The input data is not defined!")
  matOps.distributeTranspose(d.z_test)
}

private def initTargetTraining() : DenseVector[Double] = {
  assert( d.isValid() , "The input data is not defined!")
  d.z_train
}

private def initTargetTest() : DenseVector[Double] = {
  assert( d.isValid() , "The input data is not defined!")
  d.z_test
}

private def initKernelMatrixTraining() : CoordinateMatrix  = {
	assert( d.isValid() , "The input data is not defined!")
	val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_train(); value = kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
	// Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
	val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
	new CoordinateMatrix(entries, d.getN_train(), d.getN_train())
}

/**
 * Calculates the gradient vector without storing the kernel matrix Q
 *
 * In matrix notation: Q * lambda - 1
 * For an indivual entry i of the gradient vector, this is equivalent to:
 * sum over j from 1 to N of lamba(j) * d(i) * k(i,j,) * d(j) 
 * In order to avoid the constraints associated to the bias term,
 * I use a trick advocated in Christiani & Shawe-Taylor ""An Introduction to Support Vector Machines and other kernel-based learning methods"
 * 2000, pages 129-135: I add one dimension to the feature space that accounts for the bias. 
 * In input space I replace x = (x1,x2,...,xn) by x_ = (x1,x2,...,xn,tau) and w = (w1,w2,...,wn) by w_ = (w1,w2,...,wn,b/tau).
 * The SVM model <w,x>+b is then replaced by <w_,x_> = x1*w1 + x2*w2 +...+ xn*wn + tau * (b/tau).
 * In the dual formulation, I replace K(x,y) by K(x,y) + tau * tau .
 * This is not without drawbacks, because the "geometric margin of the separating hyperplane in the augmented space will typically be less
 * than that in the original space." (Christiani & Shawe-Taylor, page 131)
 * I thus have to find a suitable value of tau, which is given by the maximum of the squared euclidean norm of all inputs.
 * With this choice, it is guaranteed that the fat-shattering dimension will not increase by more than factor 4 compared to input space.
 *  
 * alphas: the current dual variables
 * tau: the a priori chosen bias term in feature space 
 * */
def calculateGradient(alphas : DenseVector[Double], tau : Double) : DenseVector[Double]  = {
  val tauSquare = tau * tau
  val N = d.getN_train()
  var v = DenseVector.fill(N){-1.0}
  for (i <- 0 until N; j <- 0 until N){
    v(i) += alphas(j) * d.z_train(i) * d.z_train(j) * (kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t) + tauSquare)
  }
  v
}

private def initKernelMatrixTest() : CoordinateMatrix = {
	assert( d.isValid() , "The input data is not defined!")
  val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_test(); value = kf.kernel(d.X_train(i,::).t, d.X_test(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
  // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
  val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
  return new CoordinateMatrix(entries, d.getN_train(), d.getN_test())
}
}

