package SVM
import SVM.DataSetType.{Test, Train}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.sql.{Dataset, SparkSession}
import breeze.stats._

trait Data{

  //Get row with row index (starting with 0) from test set.
  def getRowTest(rowIndex: Int):DenseVector[Double]

  //Get row with row index (starting with 0) from training set.
  def getRowTrain(rowIndex: Int):DenseVector[Double]

  //Get label with row index (starting with 0) from test set.
  def getLabelTest(rowIndex: Int): Double

  //Get label with row index (starting with 0) from training set.
  def getLabelTrain(rowIndex: Int): Double

  //Get vector of labels from test set.
  def getLabelsTest : DenseVector[Int]

  //Get vector of labels from training set.
  def getLabelsTrain : DenseVector[Int]

  //Was the data set correctly initialized?
  def isDefined : Boolean
  def getN_train : Int
  def getN_test : Int
  def getd : Int
}

abstract class basicDataSetEntry{
  val rowNr : Int
  val label : Int
  val x1 : Double
  val x2 : Double
  //return an ordered vector of all the predictors for a given row
  def getPredictors() :DenseVector[Double]
  def getLabel() : Int = label
  def getD() : Int
}

class SparkDataSet[T <: basicDataSetEntry](dataSetTrain: Dataset[T], dataSetTest: Dataset[T]) extends Data{

  def getRow(rowIndex: Int, dataSetType: DataSetType.Value): T = {
    if(dataSetType == Test) return dataSetTest.filter(x => x.rowNr == rowIndex).first()
    if(dataSetType == Train) return dataSetTrain.filter(x => x.rowNr == rowIndex).first()
    else throw new Exception("Unsupported data set type!")
  }

  //Get vector of labels from test set.
  def getLabels (dataSetType: DataSetType.Value): DenseVector[Int] = {
    //I have to import implicits here to be able to extract the label from the data set.
    //https://stackoverflow.com/questions/39151189/importing-spark-implicits-in-scala#39173501
    val sparkSession = SparkSession.builder.getOrCreate()
    import sparkSession.implicits._
    if(dataSetType == Test)return new DenseVector(dataSetTest.map(x => x.label).collect())
    if(dataSetType == Train)return new DenseVector(dataSetTrain.map(x => x.label).collect())
    else throw new Exception("Unsupported data set type!")
 }

  //Get row with row index (starting with 0) from test set.
  def getRowTest(rowIndex: Int):DenseVector[Double] = getRow(rowIndex, Test).getPredictors()

  //Get row with row index (starting with 0) from training set.
  def getRowTrain(rowIndex: Int):DenseVector[Double] = getRow(rowIndex, Train).getPredictors()

  //Get label with row index (starting with 0) from test set.
  def getLabelTest(rowIndex: Int): Double = getRow(rowIndex, Test).getLabel()

  //Get label with row index (starting with 0) from training set.
  def getLabelTrain(rowIndex: Int): Double = getRow(rowIndex, Train).getLabel()

  //Get vector of labels from test set.
  def getLabelsTest : DenseVector[Int] = {
    getLabels (Test)
  }

  //Get vector of labels from training set.
  def getLabelsTrain : DenseVector[Int] = {
    getLabels(Train)
  }

  //Was the data set correctly initialized?
  def isDefined : Boolean = true

  def getN_train : Int = {
    dataSetTrain.count().toInt
  }

  def getN_test : Int = {
    dataSetTest.count().toInt
  }

  def getd : Int = {
    dataSetTest.first().getD()
  }
}

/**
  * Simulated data with given data parameters.
  * @param params
  */
class SimData (val params: DataParams) extends Data {

  def getN_train : Int = params.N_train
  def getN_test : Int = params.N_test
  def getd : Int = params.d

  //Get column with column index (starting with 0) from test set.
  override def getRowTest(columnIndex: Int): DenseVector[Double] = {
      X_test(columnIndex,::).t
  }

  //Get column with column index (starting with 0) from training set.
  override def getRowTrain(columnIndex: Int): DenseVector[Double] = {
      X_train(columnIndex,::).t
  }

  //Get label with row index (starting with 0) from test set.
  override def getLabelTest(rowIndex: Int): Double = {
      z_test(rowIndex)
  }

  //Get label with row index (starting with 0) from training set.
  override def getLabelTrain(rowIndex: Int): Double = {
      z_train(rowIndex)
  }

  //Get vector of labels from test set.
  override def getLabelsTest: DenseVector[Int] = {
      z_test
  }

  //Get vector of labels from training set.
  override def getLabelsTrain: DenseVector[Int] = {
      z_train
  }

  //Was the data set correctly initialized?
  override def isDefined : Boolean = isFilled

  var isFilled = false

  //empty data matrices for training and test set
	var X_train : DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_train, params.d)
	var X_test : DenseMatrix[Double]  = DenseMatrix.zeros[Double](params.N_test, params.d)

	//stores the maximum of the square of the euclidean norm
	//var tau : Double = 0.0

	//empty vectors for the labels of training and test set
	var z_train : DenseVector[Int] = DenseVector.zeros[Int](params.N_train)
	var z_test : DenseVector[Int] =  DenseVector.zeros[Int](params.N_test)

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Synthetic dataset with "+params.d+" variables.\n")
		sb.append("Observations: "+ params.N_train +" (training), " + params.N_test+ "(test)\n")
		if(isFilled)sb.append("Data was already generated.\n")
		else sb.append("Data was not yet generated.\n")
		sb.toString()
	} 

	def simulate() : Unit = {

      // Set locations of two modes, theta1 and theta2
      val theta1 : DenseVector[Double] = DenseVector.rand(params.d)
      val theta2 : DenseVector[Double] = DenseVector.rand(params.d)

      //Randomly allocate observations to each class (0 or 1)
      val z : DenseVector[Int] = DenseVector.rand(params.N).map(x=>2*x).map(x=>floor(x)).map(x=>((2*x)-1).toInt)
      z_train = z(0 until params.N_train)
      z_test = z(params.N_train until params.N)

      val mvn1 = breeze.stats.distributions.MultivariateGaussian(theta1, diag(DenseVector.fill(params.d){1.0}))
      val mvn2 = breeze.stats.distributions.MultivariateGaussian(theta2, diag(DenseVector.fill(params.d){1.0}))

      // Simulate each observation depending on the component its been allocated
      // Create all inputs (predictors)
      var x = DenseVector.zeros[Double](params.d)

      for(i <- 0 until params.N){
        if ( z(i) == -1 ) {
            x = mvn1.sample()
        }else{
            x = mvn2.sample()
        }
        //Matrix assignment to row
        //For training set:
        if( i < params.N_train ){
          //calculate the L1 norm for vector x:
          //val norm = sqrt(x.map(e => abs(e)).reduce(_+_))
          //update tau as maximum over all norms
          //tau = max(tau, norm)
          X_train(i, ::) := DenseVector(x.toArray).t
        //For the tests set:
        }else{
          X_test(i - params.N_train, ::) := DenseVector(x.toArray).t
        }
      }
      isFilled = true
    }
  }

/**
  * Data that is stored in the local file system as csv files.
  */
class LocalData extends Data{

  /**
    * Matrix with 3 rows for mean, variance, and standard deviation
    * and columns for data columns
    */
  var trainSummary = DenseMatrix.zeros[Double](3,1)

    //Get column with column index (starting with 0) from test set.
    override def getRowTest(columnIndex: Int): DenseVector[Double] = {
      X_test(columnIndex,::).t
    }

    //Get column with column index (starting with 0) from training set.
    override def getRowTrain(columnIndex: Int): DenseVector[Double] = {
      X_train(columnIndex,::).t
    }

    //Get label with row index (starting with 0) from test set.
    override def getLabelTest(rowIndex: Int): Double = {
      z_test(rowIndex)
    }

    //Get label with row index (starting with 0) from training set.
    override def getLabelTrain(rowIndex: Int): Double = {
      z_train(rowIndex)
    }

    //Get vector of labels from test set.
    override def getLabelsTest: DenseVector[Int] = {
      z_test
    }

    //Get vector of labels from training set.
    override def getLabelsTrain: DenseVector[Int] = {
      z_train
    }

    //Was the data set correctly initialized?
    override def isDefined : Boolean = isFilled

    var N : Int = 0
    var N_train : Int = 0
    var N_test : Int = 0
    var d : Int = 0

    var isFilled = false
    var testSetIsFilled = false
    var trainingSetIsFilled = false

    //empty data matrices for training and test set
    var X_train, X_test : DenseMatrix[Double] = DenseMatrix.zeros[Double](1, 1)

    //empty vectors for the labels of training and test set
    var z_train, z_test : DenseVector[Int] = DenseVector.zeros[Int](1)

    override def toString : String = {
      val sb = new StringBuilder
      sb.append("Empirical dataset from local file system with "+ d+" variables.\n")
      sb.append("Observations: "+ N_train +" (training), " + N_test+ " (test)\n")
      if(isFilled) sb.append("Data was already generated.\n")
      else sb.append("Data was not yet generated.\n")
      sb.toString()
    }

    def readTrainingDataSet (path: String, separator: Char, columnIndexClass: Int) : Unit = {
      val csvReader : CSVReader = new CSVReader(path, separator, columnIndexClass)
      val (inputs, labels) = csvReader.read()
      X_train = inputs
      trainSummary = summary(Train)
      println("Summary statistics of train data set before z-transformation:")
      println(trainSummary.t)
      //TODO Do z-transformation of X_train!
      val means = columnMeans(X_train)
      val stdev = columnStandardDeviation(X_train)
      val N = X_train.rows
      val Means = tile(means, 1, N).t
      val SD = tile(stdev, 1, N).t

      X_train = (X_train - Means) / SD
      z_train = labels
      d = X_train.cols
      N_train = X_train.rows
      trainingSetIsFilled = true
      isFilled = testSetIsFilled && trainingSetIsFilled
    }

    def readTestDataSet (path: String, separator: Char, columnIndexClass: Int) : Unit = {
      val csvReader : CSVReader = new CSVReader(path, separator, columnIndexClass)
      val (inputs, labels) = csvReader.read()
      X_test = inputs
      val testSummary = summary(Test)
      println("Summary statistics of test data set before z-transformation with means and standard deviation of the training set:")
      println(testSummary.t)
      //TODO Do z-transformation of X_test with means and SDev of X_train!
      val means = columnMeans(X_train)
      val stdev = columnStandardDeviation(X_train)
      val N = X_test.rows
      val Means = tile(means, 1, N).t
      val SD = tile(stdev, 1, N).t
      X_train = (X_train - Means) / SD
      z_test = labels
      d = X_test.cols
      N_test = X_test.rows
      testSetIsFilled = true
      isFilled = testSetIsFilled && trainingSetIsFilled
    }

  /**
    * @param dataSetType
    * @return
    */
  def summary(dataSetType: DataSetType.Value):DenseMatrix[Double]={
    if(dataSetType == Test) return createSummary(X_test)
    if(dataSetType == Train) return createSummary(X_train)
    else throw new Exception("Unsupported data set type!")
  }

  def columnMeans(m: DenseMatrix[Double]):DenseVector[Double] = mean(m(::,*)).t
  def columnVariance(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t
  def columnStandardDeviation(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t.map(x=>sqrt(x))

  /**
    * Returns column summary statistics (mean, var, standard deviation) as a matrix.
    * @param m
    * @return
    */
  def createSummary(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val summaryM = DenseMatrix.zeros[Double](3,m.cols)
    summaryM.t(::,0) := columnMeans(m)
    summaryM.t(::,1) := columnVariance(m)
    summaryM.t(::,2) := columnStandardDeviation(m)
    summaryM
  }

  override def getN_train: Int = N_train

  override def getN_test: Int = N_test

  override def getd: Int = d
}
