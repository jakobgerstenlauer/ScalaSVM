package SVM

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import SVM.DataSetType.{Test, Train}

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

