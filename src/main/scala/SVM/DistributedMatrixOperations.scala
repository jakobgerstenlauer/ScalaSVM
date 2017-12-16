package SVM

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

case class allAlphasZeroException(smth:String) extends Exception(smth)

class DistributedMatrixOps(sc: SparkContext){

  /**
  * Prints all elements of a CoordinateMatrix to the console.
  **/
  def print(m: CoordinateMatrix): Unit = {
    m.entries.foreach(println)
  }

  /**
  * Prints the first row of a CoordinateMatrix to the console.
  **/
  def printFirstRow(m: CoordinateMatrix): Unit = {
    m.entries.filter({case MatrixEntry(i,_,_) => if(i==0) true else false}).foreach(println)
  }


  def fill(v: DenseVector[Double], index: Int, value: Double) : Unit = {
    v(index)=value
  }

  /**
  * Transform a CoordinateMatrix with one column into a local column vector.
  **/
  def collectColumnVector(m: CoordinateMatrix): DenseVector[Double] = {
    assert(m.numCols() == 1, "Coordinate matrix is empty or has more than one column!")
    val rows: Int = m.numRows().toInt
    var v: DenseVector[Double] = DenseVector.zeros[Double](rows)
    m.entries.collect foreach { case MatrixEntry(index, _, value) => fill(v, index.toInt, value) }
    v
  }

  /**
  * Transform a CoordinateMatrix with one row into a local column vector.
  **/
  def collectRowVector(m: CoordinateMatrix): DenseVector[Double] = {
    assert(m.numRows() == 1, "Coordinate matrix is empty or has more than one row!")
    val columns = m.numCols().toInt
    var v: DenseVector[Double] = DenseVector.zeros[Double](columns)
    m.entries.collect foreach {case MatrixEntry(_, index, value) => fill(v, index.toInt, value) }
    v
  }

  /**
  * Prints the nth row (starting with 0) of a CoordinateMatrix to the console.
  **/
  def printRow(m: CoordinateMatrix, row: Int): Unit = {
      m.entries.map({case MatrixEntry(i,j,v) => if(i==row) println("col:"+j+":"+v)})
  }

  //Source:
  //https://www.balabit.com/blog/scalable-sparse-matrix-multiplication-in-apache-spark/
  def coordinateMatrixMultiply(leftMatrix: CoordinateMatrix, rightMatrix: CoordinateMatrix): CoordinateMatrix = {
    assert(leftMatrix.numCols()==rightMatrix.numRows(),"The number of columns of the left matrix does not equal the number of rows of the right matrix!")

    val M_ = leftMatrix.entries.map({ case MatrixEntry(i, j, v) => (j, (i, v)) })
    val N_ = rightMatrix.entries.map({ case MatrixEntry(j, k, w) => (j, (k, w)) })

    val productEntries = M_
      .join(N_)
      .map({ case (_, ((i, v), (k, w))) => ((i, k), v * w) })
      .reduceByKey(_ + _)
      .map({ case ((i, k), sum) => MatrixEntry(i, k, sum) })
    new CoordinateMatrix(productEntries,leftMatrix.numRows(),leftMatrix.numCols())
  }

  def coordinateMatrixSignumAndMultiply(leftMatrix: CoordinateMatrix, rightVector: CoordinateMatrix): CoordinateMatrix = {
      assert(leftMatrix.numCols()==rightVector.numRows(),"The number of columns of the left matrix does not equal the number of rows (=entries) of the right matrix!")
    assert(rightVector.numCols()==1,"The right matrix must be a vector!")

    val M_ = leftMatrix.entries.map({ case MatrixEntry(i, j, v) => (j, (i, signum(v))) })
    val N_ = rightVector.entries.map({ case MatrixEntry(j, k, w) => (j, (k, w)) })

    val productEntries: RDD[MatrixEntry] = M_
      .join(N_)
      .map({ case (_, ((i, v), (_, w))) => ((i, 0), v * w) })
      .reduceByKey(_ + _)
      .map({ case ((i, 0), sum) => MatrixEntry(i, 0, sum) })
    new CoordinateMatrix(productEntries,leftMatrix.numRows(),1)
  }

  def createEmptyCoordinateMatrix(numRows: Int, numCols: Int) : CoordinateMatrix={
    assert(numRows > 0, "The number of rows must be positive!")
    assert(numCols > 0, "The number of columns must be positive!")
    val listOfMatrixEntries =  List(MatrixEntry(0, 0, 0))
    new CoordinateMatrix(sc.parallelize(listOfMatrixEntries), numRows, numCols)
  }

  def distributeRowVector(a: DenseVector[Double], epsilon: Double) : CoordinateMatrix={
      assert(epsilon > 0, "The value of epsilon must be positive!")
      assert(a.length > 0, "The input vector with the alphas is empty!!!")

      def exceeds(x: Double, e: Double) : Boolean = abs(x) > e

      val listOfMatrixEntries =  for (i <- 0 until a.length) yield MatrixEntry(0, i, a(i))
      // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
      val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries.filter(x => exceeds(x.value,epsilon)))

      if(entries.count()==0){
        throw allAlphasZeroException("All values of the distributed row vector are zero!")
      }

      // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
      new CoordinateMatrix(entries, 0, a.length.toLong)
  }

  def distributeColumnVector(a: DenseVector[Double], epsilon: Double) : CoordinateMatrix={
      assert(epsilon > 0, "The value of epsilon must be positive!")
      assert(a.length > 0, "The input vector with the alphas is empty!!!")

      def exceeds(x: Double, e: Double) : Boolean = abs(x) > e

      val listOfMatrixEntries =  for (i <- 0 until a.length) yield MatrixEntry(i, 0, a(i))
      // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
      val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries.filter(x => exceeds(x.value,epsilon)))
      // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].

      if(entries.count()==0){
        throw allAlphasZeroException("All values of the distributed column vector are zero!")
      }

      new CoordinateMatrix(entries, a.length.toLong, 0)
  }

  //Transform a local dense vector into a distributed coordinate matrix.
  def distribute(a: DenseVector[Double]) : CoordinateMatrix={
      assert(a.length>0, "The length of the input vector is zero.")
      val listOfMatrixEntries =  for (i <- 0 until a.length) yield MatrixEntry(0, i, a(i))
      // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
      val entriesTest: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
     // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
     new CoordinateMatrix(entriesTest, 1, a.length.toLong)
  }

  //Transform a local dense vector into a transposed, distributed coordinate matrix.
  def distributeTranspose(a: DenseVector[Int]) : CoordinateMatrix={
      assert(a.length>0, "The length of the input vector is zero.")
      val listOfMatrixEntries =  for (i <- 0 until a.length) yield MatrixEntry(i, 0, a(i))
      // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
      val entriesTest: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
      // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
      new CoordinateMatrix(entriesTest, a.length.toLong, 1)
  }

  def getRow(m: CoordinateMatrix, rowIndex: Int): Option[DenseVector[Double]]={
      assert(m!= null, "The input matrix is not defined!")
      assert(m.numRows()-1>=rowIndex, "The row index is higher than the number of rows of the input matrix!")
      assert(rowIndex>=0, "The row index must be positive!")
      val values = m.toIndexedRowMatrix.rows.filter(_.index == rowIndex).map(x => x.vector.toArray).collect()
      try{
          Some(new DenseVector(values(0)))
      } catch {
          case e: Exception => None
      }
  }
}
