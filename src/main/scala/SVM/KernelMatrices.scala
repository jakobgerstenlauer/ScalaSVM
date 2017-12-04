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
 * Calculate the gradient without storing the kernel matrix Q: 
 * In matrix notation: Q * lambda - 1
 * For an indivual entry i of the gradient vector, this is equivalent to:
 * sum over j from 1 to N of lamba(j) * d(i) * k(i,j,) * d(j) - 1
 * */
def calculateGradient(alphas : DenseVector[Double]) : DenseVector[Double]  = {
  val N = d.getN_train()
  var v = DenseVector.fill(N){-1.0}
  for (i <- 0 until N; j <- 0 until N){
    v(i) += alphas(j) * d.z_train(i) * d.z_train(j) * kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t)
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

