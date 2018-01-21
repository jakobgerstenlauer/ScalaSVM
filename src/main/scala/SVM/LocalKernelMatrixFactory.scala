package SVM

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

trait MatrixFactory{
  def calculateGradient(alpha: DenseVector[Double]):DenseVector[Double]
  def getData:Data
  def predictOnTrainingSet(alphas : DenseVector[Double]):DenseVector[Double]
  def predictOnTestSet(alphas : DenseVector[Double]):DenseVector[Double]
}

abstract class BaseMatrixFactoryWithMatrices(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){
  val z : DenseVector[Int] = initTargetTraining()
  val z_validation : DenseVector[Int] = initTargetValidation()

  private def initTargetTraining() : DenseVector[Int] = {
    assert(d.isDefined, "The input data is not defined!")
    d.getLabelsTrain
  }

  private def initTargetValidation() : DenseVector[Int] = {
    assert(d.isDefined, "The input data is not defined!")
    d.getLabelsValidation
  }
}

case class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext) extends BaseMatrixFactoryWithMatrices(d, kf, epsilon) {

  val K  = initKernelMatrixTraining()
  val S  = initKernelMatrixValidation()

  def initKernelMatrixTraining() : CoordinateMatrix  = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_Train; j <- 0 until d.getN_Train;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_Train, d.getN_Train)
  }

  def initKernelMatrixValidation () : CoordinateMatrix = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_Train; j <- 0 until d.getN_Validation;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowValidation(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_Train, d.getN_Validation)
  }
}

case class LocalKernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactoryWithMatrices(d, kf, epsilon){

  val K  = initKernelMatrixTraining()
  val S  = initKernelMatrixValidation()

  def initKernelMatrixTraining() : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_Train, d.getN_Train)
    for (i <- 0 until d.getN_Train; j <- 0 until d.getN_Train; value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))) K(i, j)=value
    K
  }

  def initKernelMatrixValidation () : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_Train, d.getN_Validation)
    for (i <- 0 until d.getN_Train; j <- 0 until d.getN_Validation; value = kf.kernel(d.getRowTrain(i), d.getRowValidation(j))) K(i, j)=value
    K
  }
}

