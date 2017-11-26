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

class KernelMatrixFactory(val d: Data, val kf: KernelFunction, val epsilon: Double, val sc: SparkContext){

val matOps = new DistributedMatrixOps(sc)
val K = initKernelMatrixTraining()
//val Q = initQMatrixTraining()
val S = initKernelMatrixTest()
val Z = initTargetMatrixTraining()
val Z_test = initTargetMatrixTest()
//val rowColumnIndexMap = initRowColumnIndexMap()

def getData() : Data = return d

private def initTargetMatrixTraining() : CoordinateMatrix = {
        assert( d.isValid() , "The input data is not defined!")
	return matOps.distributeTranspose(d.z_train)
}

private def initTargetMatrixTest() : CoordinateMatrix = {
        assert( d.isValid() , "The input data is not defined!")
        return matOps.distributeTranspose(d.z_test)
}

private def initKernelMatrixTraining() : CoordinateMatrix  = {
	assert( d.isValid() , "The input data is not defined!")
	val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_train(); value = kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
	// Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
	val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
	return new CoordinateMatrix(entries, d.getN_train(), d.getN_train())
}

//private def initRowColumnIndexMap() : collection.mutable.MultiMap[Long, Set[Long]]  = {
//	assert( d.isValid() , "The input data is not defined!")
//        val indexMap = new HashMap[Long, Set[Long]] with MultiMap[Long, Long]
//	for (i <- 0 until d.getN_train(); j <- 0 until d.getN_train(); value = kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t); if (value > epsilon))
//		mm.addBinding(i.toLong, j.toLong)
//	return indexMap
//}


//private def initQMatrixTraining() : CoordinateMatrix  = {
//	assert( d.isValid() , "The input data is not defined!")
//	val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_test(); value = d.z_train(i) * d_z_train(j) * kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
//	// Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
//	val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
//	return new CoordinateMatrix(entries, d.getN_train(), d.getN_train())
//}

//TODO: sum the elements with identical column index j and subtract 1 to get the gradient!
//private def calculateGradient() : DenseVector[Double]  = {
//	assert( d.isValid() , "The input data is not defined!")
//        for (pairs <- listOfIndexPairs; val i = pairs.i; val j = pairs.j; value = alpha(j) * d.z_train(i) * d_z_train(j) * kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t)) yield (new MatrixEntry(i, j, value))
//}

private def initKernelMatrixTest() : CoordinateMatrix = {
	assert( d.isValid() , "The input data is not defined!")
        val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_test(); value = kf.kernel(d.X_train(i,::).t, d.X_test(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
        // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
        val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
        return new CoordinateMatrix(entries, d.getN_train(), d.getN_test())
}

}

