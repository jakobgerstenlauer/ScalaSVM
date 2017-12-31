package SVM

import breeze.linalg._

trait MatrixFactory{
  def calculateGradient(alpha: DenseVector[Double]):DenseVector[Double]
  def getData():Data
  def predictOnTrainingSet(alphas : DenseVector[Double]):DenseVector[Double]
  def predictOnTestSet(alphas : DenseVector[Double]):DenseVector[Double]
}

case class LocalKernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){

  val K : DenseMatrix[Double] = initKernelMatrixTraining()
  val S : DenseMatrix[Double] = initKernelMatrixTest()
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

  private def initKernelMatrixTraining() : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_train, d.getN_train)
    for (i <- 0 until d.getN_train; j <- 0 until d.getN_train; value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))) K(i, j)=value
    K
  }

  private def initKernelMatrixTest() : DenseMatrix[Double]  = {
    assert(d.isDefined, "The input data is not defined!")
    val K : DenseMatrix[Double] = DenseMatrix.zeros[Double](d.getN_train, d.getN_test)
    for (i <- 0 until d.getN_train; j <- 0 until d.getN_test; value = kf.kernel(d.getRowTrain(i), d.getRowTest(j))) K(i, j)=value
    K
  }
}

