package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps 
import breeze.linalg.operators 
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext){

def getData() : Data = return d

def getKernelMatrixTraining() : CoordinateMatrix={
	assert( d.isValid() , "The input data is not defined!")
	val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_train(); value = kf.kernel(d.X_train(i,::).t, d.X_train(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
	// Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
	val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
	return new CoordinateMatrix(entries, d.getN_train(), d.getN_train())
}

def getKernelMatrixTest() : CoordinateMatrix={
	assert( d.isValid() , "The input data is not defined!")
        val listOfMatrixEntries =  for (i <- 0 until d.getN_train(); j <- 0 until d.getN_test(); value = kf.kernel(d.X_train(i,::).t, d.X_test(j,::).t); if (value > epsilon)) yield (new MatrixEntry(i, j, value))
        // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
        val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
        return new CoordinateMatrix(entries, d.getN_train(), d.getN_test())
}

}

