package SVM

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import SVM.DataSetType.{Test, Train}
import scala.collection.mutable.{HashMap, MultiMap}

trait MatrixFactory{
  def calculateGradient(alpha: DenseVector[Double]):DenseVector[Double]
  def getData():Data
  def predictOnTrainingSet(alphas : DenseVector[Double]):DenseVector[Double]
  def predictOnTestSet(alphas : DenseVector[Double]):DenseVector[Double]
}

abstract class BaseMatrixFactoryWithMatrices(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){
  val z : DenseVector[Int] = initTargetTraining()
  val z_test : DenseVector[Int] = initTargetTest()

  private def initTargetTraining() : DenseVector[Int] = {
    assert(d.isDefined, "The input data is not defined!")
    d.getLabelsTrain
  }

  private def initTargetTest() : DenseVector[Int] = {
    assert(d.isDefined, "The input data is not defined!")
    d.getLabelsTest
  }


  /**
    * Returns the number of non-sparse matrix elements for a given epsilon
    * @param typeOfMatrix For the training matrix K or the test matrix S?
    * @param epsilon Value below which similarities will be ignored.
    * @return
    */
  def probeSparsity(typeOfMatrix: DataSetType.Value, epsilon: Double): Int = {
    typeOfMatrix match{
      case Train => probeSparsityK(epsilon)
      case Test => probeSparsityS(epsilon)
      case _ => throw new Exception("Unsupported data set type!")
    }
  }

  /**
    * Returns the number of non-sparse matrix elements for a given epsilon and the training matrix K.
    * @param epsilon Value below which similarities will be ignored.
    * @return
    */
  def probeSparsityK(epsilon: Double): Int = {
    val N = d.getN_train
    var size2 : Int = N //the diagonal elements are non-sparse!
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N; j <- (i+1) until N if(kf.kernel(d.getRowTrain(i), d.getRowTrain(j)) > epsilon)){
      size2 = size2 + 2
    }
    size2
  }

  /**
    * Returns the number of non-sparse matrix elements for a given epsilon and the training vs test matrix S.
    * @param epsilon
    * @return
    */
  def probeSparsityS(epsilon: Double): Int = {
    var size2 : Int = 0
    val N_train = d.getN_train
    val N_test = d.getN_test
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N_test; j <- (i+1) until N_train
         if(kf.kernel(d.getRowTest(i), d.getRowTrain(j)) > epsilon)){
      size2 = size2 + 2
    }
    //iterate over the diagonal
    for (i <- 0 until max(N_test,N_train)
        if(kf.kernel(d.getRowTest(i), d.getRowTrain(i)) > epsilon)){
      size2 = size2 + 1
    }
    size2
  }
}

case class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext) extends BaseMatrixFactoryWithMatrices(d, kf, epsilon) {

  val K  = initKernelMatrixTraining()
  val S  = initKernelMatrixTest()

  def initKernelMatrixTraining() : CoordinateMatrix  = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_train; j <- 0 until d.getN_train;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_train, d.getN_train)
  }

  def initKernelMatrixTest() : CoordinateMatrix = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_train; j <- 0 until d.getN_test;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowTest(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_train, d.getN_test)
  }
}

case class LocalKernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactoryWithMatrices(d, kf, epsilon){

  val K  = initKernelMatrixTraining()
  val S  = initKernelMatrixTest()

  def initKernelMatrixTraining() : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_train, d.getN_train)
    for (i <- 0 until d.getN_train; j <- 0 until d.getN_train; value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))) K(i, j)=value
    K
  }

  def initKernelMatrixTest() : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_train, d.getN_test)
    for (i <- 0 until d.getN_train; j <- 0 until d.getN_test; value = kf.kernel(d.getRowTrain(i), d.getRowTest(j))) K(i, j)=value
    K
  }
}

