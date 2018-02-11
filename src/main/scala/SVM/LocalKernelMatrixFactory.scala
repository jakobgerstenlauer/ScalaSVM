package SVM

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import SVM.DataSetType.{Test, Train, Validation}
import breeze.numerics.signum
import scala.concurrent.{Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global

trait MatrixFactory{
  def calculateGradient(alpha: DenseVector[Double]):DenseVector[Double]
  def getData:Data
  def predictOnTrainingSet(alphas : DenseVector[Double]):DenseVector[Double]
  def predictOnValidationSet(alphas : DenseVector[Double]):DenseVector[Double]
  def predictOnTestSet(alphas : DenseVector[Double]):DenseVector[Double]
}

abstract class BaseMatrixFactoryWithMatrices(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){
  val z : DenseVector[Int] = initTarget(Train)
  val z_validation : DenseVector[Int] = initTarget(Validation)
  val z_test : DenseVector[Int] = initTarget(Test)
  private def initTarget(dataType : SVM.DataSetType.Value) : DenseVector[Int] = {
    assert(d.isDefined, "The input data is not defined!")
    d.getLabels(dataType)
  }
}

case class KernelMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double, sc: SparkContext) extends BaseMatrixFactoryWithMatrices(d, kf, epsilon) {

  private val matOps = new DistributedMatrixOps(sc)

  /**
    * Q matrix for training set
    */
  val Q = initQMatrix()

  /**
    * Kernel matrix for validation set
    */
  val V = initKernelMatrix(Validation)

  /**
    * Kernel matrix for test set
    */
  val T  = initKernelMatrix(Test)

  def getKernelMatrix(dataType : SVM.DataSetType.Value): CoordinateMatrix = {
    dataType match {
      case Validation => this.V
      case Test => this.T
    }
  }

  def initQMatrix() : CoordinateMatrix = {
    assert(d.isDefined, "The input data is not defined!")
    val numCols = d.getN(Train)
    val listOfMatrixEntries =  for (i <- 0 until d.getN_Train; label_i = z(i); j <- 0 until numCols;
                                    value = label_i * z(j) * kf.kernel(d.getRow(Train,i), d.getRow(Train,j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    val m = new CoordinateMatrix(entries, d.getN_Train, numCols)
    println("Q matrix has rows: "+ m.numRows() +" and columns: "+ m.numCols())
    m
  }

  def initKernelMatrix(dataType : SVM.DataSetType.Value) : CoordinateMatrix = {
    assert(d.isDefined, "The input data is not defined!")
    val numCols = d.getN(dataType)
    val listOfMatrixEntries =  for (i <- 0 until d.getN_Train; j <- 0 until numCols;
                                    value = kf.kernel(d.getRow(Train,i), d.getRow(dataType,j))
                                    if value > epsilon) yield MatrixEntry(i, j, value)
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries)
    val m = new CoordinateMatrix(entries, d.getN_Train, numCols)
    println("Kernel matrix of type "+dataType.toString()+" has rows: "+ m.numRows() +" and columns: "+ m.numCols())
    m
  }

  override def calculateGradient(alphas: DenseVector[Double]) : DenseVector[Double]  = {
    val A = matOps.distributeColumnVector(alphas, epsilon)
    val P = matOps.coordinateMatrixMultiply(Q, A)
    DenseVector.fill[Double](alphas.length){-1} + matOps.collectColumnVector(P)
  }

  /**
    * Calculates the number of correctly and incorrectly classified instances as tuple.
    * @param predictions The vector of class predictions.
    * @param labels The vector of empirical classes.
    * @return A tuple consisting of first the number of correct predictions (classifications)
    *         and second the number of misclassifications.
    */
  def calculateAccuracy(predictions: DenseVector[Double], labels: DenseVector[Int]):Int = {
    assert(predictions.length == labels.length, "Length predictions:"+predictions.length+"!= length labels:"+labels.length)
    val product : DenseVector[Double] = predictions *:* labels.map(x => x.toDouble)
    product.map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  def evaluate(alphas: Alphas, ap: AlgoParams, kmf: KernelMatrixFactory, dataType: SVM.DataSetType.Value, iteration: Int):Future[(Int,Int)]= {
    val promise = Promise[(Int,Int)]
    Future{
      val K : CoordinateMatrix = kmf.getKernelMatrix(dataType)
      val z = kmf.z.map(x=>x.toDouble)
      val epsilon = max(min(ap.epsilon, min(alphas.alpha)), 0.000001)
      val A = matOps.distributeRowVector(alphas.alpha *:* z, epsilon)
      val P = matOps.coordinateMatrixMultiply(A, K)
      val prediction = signum(matOps.collectRowVector(P))
      val correct = calculateAccuracy(prediction, d.getLabels(dataType))
      promise.success((correct, iteration))
    }
    promise.future
  }
}