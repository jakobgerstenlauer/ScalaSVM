package SVM

import breeze.linalg._
import breeze.numerics.signum
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

case class LeanMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon)

case class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext) extends BaseMatrixFactory(d, kf, epsilon) {

  val matOps = new DistributedMatrixOps(sc)
  val K : CoordinateMatrix = initKernelMatrixTraining()
  val S : CoordinateMatrix = initKernelMatrixTest()
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

  private def initKernelMatrixTraining() : CoordinateMatrix  = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_train; j <- 0 until d.getN_train;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_train, d.getN_train)
  }

 private def initKernelMatrixTest() : CoordinateMatrix = {
    assert(d.isDefined, "The input data is not defined!")
    val listOfMatrixEntries =  for (i <- 0 until d.getN_train; j <- 0 until d.getN_test;
                                    value = kf.kernel(d.getRowTrain(i), d.getRowTest(j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    new CoordinateMatrix(entries, d.getN_train, d.getN_test)
  }

  override def getData (): Data = d
}

