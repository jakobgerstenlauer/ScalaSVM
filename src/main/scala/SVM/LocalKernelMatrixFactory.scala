package SVM

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import SVM.DataSetType.{Test, Train, Validation}
import breeze.numerics.signum

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
  /**
    * Kernel matrix for training set
    */
  val K  = initKernelMatrix(Train)

  /**
    * Kernel matrix for validation set
    */
  val V  = initKernelMatrix(Validation)

  /**
    * Kernel matrix for test set
    */
  val T  = initKernelMatrix(Test)

  def getKernelMatrix(dataType : SVM.DataSetType.Value): CoordinateMatrix = {
    dataType match {
      case Train => this.K
      case Validation => this.V
      case Test => this.T
    }
  }

  def initKernelMatrix(dataType : SVM.DataSetType.Value) : CoordinateMatrix  = {
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

  def evaluate(alphas: Alphas, ap: AlgoParams, kmf: KernelMatrixFactory, matOps: DistributedMatrixOps, dataType: SVM.DataSetType.Value):DenseVector[Double]= {
    //Get the distributed kernel matrix for the given typ of data set:
    val K : CoordinateMatrix = kmf.getKernelMatrix(dataType)
    println("Kernel matrix with rows: "+K.numRows()+ "and columns: "+K.numCols() + ".")
    assert(K.numCols()>0, "The number of columns of the kernel matrix is zero.")
    assert(K.numRows()>0, "The number of rows of the kernel matrix is zero.")
    val z = kmf.z.map(x=>x.toDouble)
    println("Label vector z with length: "+z.length + ".")
    assert(K.numRows()==z.length,"The number of rows "+K.numCols()
      +"of the kernel matrix does not equal the length "+z.length+" of the vector of labels!")
    val epsilon = max(min(ap.epsilon, min(alphas.alpha)), 0.000001)
    val A = matOps.distributeRowVector(alphas.alpha *:* z, epsilon)
    assert(z!=null && A!=null && K!=null, "One of the input matrices is undefined!")
    assert(A.numCols()>0, "The number of columns of A is zero.")
    assert(A.numRows()>0, "The number of rows of A is zero.")
    assert(A.numCols()==K.numRows(),"The number of columns of A does not equal the number of rows of the kernel matrix!")
    val P = matOps.coordinateMatrixMultiply(A, K)
    println("Result matrix with rows: "+P.numRows()+ "and columns: "+P.numCols() + ".")
    //Return the predictions
    val predictions = signum(matOps.collectRowVector(P))
    println("Predictions vector with length: "+predictions.length + ".")
    predictions
  }
}