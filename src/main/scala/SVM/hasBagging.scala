package SVM

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

trait hasBagging extends Algorithm{

  def getDistributedAlphas(ap: AlgoParams, alphas: Alphas, kmf: MatrixFactory, sc: SparkContext) : CoordinateMatrix = {
    val batchMatrix = getBatchMatrix(ap, kmf)
    createStochasticGradientMatrix(alphas, batchMatrix, ap.epsilon, sc)
  }

  private def getBatchMatrix(ap: AlgoParams, kmf: MatrixFactory) : DenseMatrix[Double] = {
    val randomMatrix : DenseMatrix[Double] = DenseMatrix.rand(ap.numBaggingReplicates, kmf.getData().getN_train)
    val batchMatrix = randomMatrix.map(x=>if(x < ap.batchProb) 1.0 else 0.0)
    batchMatrix
  }

  private	def exceeds(x: Double, e: Double) : Boolean = abs(x) > e

  private def createStochasticGradientMatrix(alphas: Alphas, m: DenseMatrix[Double], epsilon: Double, sc: SparkContext) : CoordinateMatrix = {
    val a = alphas.alpha
    val a_old = alphas.alphaOld
    assert(epsilon > 0, "The value of epsilon must be positive!")
    assert(a.length > 0, "The input vector with the alphas is empty!!!")
    assert(m.rows > 0, "The dense matrix m must have at least 1 row!!!")
    assert(m.cols == a.length, "The number of columns of the matrix m("+m.cols+") must be equal to the length of alpha("+a.length+")!!!")
    //If entry i,j of the matrix m is 1 we set element i,j of A to a else a_old:
    val listOfMatrixEntries =  for (i <- 0 until m.rows; j <- 0 until a.length) yield MatrixEntry(i, j, m(i,j)*a(j)+(1-m(i,j))*a_old(j))
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries.filter(x => exceeds(x.value,epsilon)))
    //entries.collect().map({ case MatrixEntry(row, column, value) => println("row: "+row+" column: "+column+" value: "+value)})
    if(entries.count()==0) throw allAlphasZeroException("All values of the distributed matrix are zero!")
    // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
    new CoordinateMatrix(entries, m.rows, a.length.toLong)
  }
}
