package SVM
import SVM.DataSetType.{Test, Train, Validation}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.sql.{Dataset, SparkSession}
import breeze.stats._

trait Data{
  def getRow(dataSetType: DataSetType.Value, rowIndex: Int) : DenseVector[Double]
  def getLabel(dataSetType: DataSetType.Value, rowIndex: Int) : Double
  def getLabels(dataSetType: DataSetType.Value) : DenseVector[Int]
  //Was the data set correctly initialized?
  def isDefined : Boolean
  def getN_Train : Int
  def getN_Validation : Int
  def getN_Test : Int
  def getd : Int
  //print distribution of labels in validation and training set
  def tableLabels(): Unit
  def tableLabels(vector: DenseVector[Int], tag: String): Unit = {
    val positive = vector.map(x => if (x > 0) 1 else 0).reduce(_ + _)
    val negative = vector.map(x => if (x < 0) 1 else 0).reduce(_ + _)
    println(tag +": " + positive + "(+)/" + negative + "(-)/" + vector.length + "(total)")
  }
}

abstract class basicDataSetEntry{
  val rowNr : Int
  val label : Int
  val x1 : Double
  val x2 : Double
  //return an ordered vector of all the predictors for a given row
  def getPredictors:DenseVector[Double]
  def getLabel: Int = label
  def getD: Int
}

class SparkDataSet[T <: basicDataSetEntry](dataSetTrain: Dataset[T], dataSetValidation: Dataset[T], dataSetTest: Dataset[T]) extends Data{

  def tableLabels(): Unit = {
    val z_train = getLabels(Train)
    val z_validation = getLabels(Validation)
    val z_test = getLabels(Test)
    tableLabels(z_train, "Training")
    tableLabels(z_validation, "Validation")
    tableLabels(z_test, "Test")
  }

  def getSparkRow (dataSetType: DataSetType.Value, rowIndex: Int): T = {
    dataSetType match{
      case Validation => dataSetValidation.filter(x => x.rowNr == rowIndex).first()
      case Train => dataSetTrain.filter(x => x.rowNr == rowIndex).first()
      case Test => dataSetTest.filter(x => x.rowNr == rowIndex).first()
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  def getLabels (dataSetType: DataSetType.Value): DenseVector[Int] = {
    //I have to import implicits here to be able to extract the label from the data set.
    //https://stackoverflow.com/questions/39151189/importing-spark-implicits-in-scala#39173501
    val sparkSession = SparkSession.builder.getOrCreate()
    import sparkSession.implicits._
    dataSetType match{
      case Validation => new DenseVector(dataSetValidation.map(x => x.label).collect())
      case Train => new DenseVector(dataSetTrain.map(x => x.label).collect())
      case Test => new DenseVector(dataSetTest.map(x => x.label).collect())
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }
  def getLabel(dataSetType: DataSetType.Value, rowIndex: Int):Double = getSparkRow(dataSetType, rowIndex).getLabel
  def getRow (dataSetType: DataSetType.Value, rowIndex: Int): DenseVector[Double] = getSparkRow(dataSetType, rowIndex).getPredictors
  def isDefined : Boolean = true
  def getN_Train : Int = dataSetTrain.count().toInt
  def getN_Validation : Int = dataSetValidation.count().toInt
  def getN_Test : Int = dataSetTest.count().toInt
  def getd : Int = dataSetValidation.first().getD
}

abstract class LData extends Data {

  //empty data matrices for training and validation set
  var X_train: DenseMatrix[Double]
  var X_validation: DenseMatrix[Double]
  var X_test: DenseMatrix[Double]

  //empty vectors for the labels of training and validation set
  var z_train : DenseVector[Int]
  var z_validation : DenseVector[Int]
  var z_test : DenseVector[Int]

  //Get column with column index (starting with 0) from validation set.
  override def getRow (dataSetType: DataSetType.Value, columnIndex: Int): DenseVector[Double] = {
    dataSetType match{
      case Validation => X_validation(columnIndex, ::).t
      case Train => X_train(columnIndex, ::).t
      case Test => X_test(columnIndex, ::).t
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  override def getLabel (dataSetType: DataSetType.Value, rowIndex: Int): Double = {
    dataSetType match{
      case Validation => z_validation(rowIndex)
      case Train => z_train(rowIndex)
      case Test => z_test(rowIndex)
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  override def getLabels (dataSetType: DataSetType.Value): DenseVector[Int] = {
    dataSetType match{
      case Validation => z_validation
      case Train => z_train
      case Test => z_test
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  def tableLabels(): Unit = {
    tableLabels(z_train, "Training")
    tableLabels(z_validation, "Validation")
    tableLabels(z_test, "Test")
  }
}

/**
  * Simulated data with given data parameters.
  * @param params
  */
class SimData (val params: DataParams) extends LData {
  def getN_Train : Int = params.N_train
  def getN_Validation : Int = params.N_validation
  def getN_Test : Int = params.N_test
  def getd : Int = params.d

  //Was the data set correctly initialized?
  override def isDefined : Boolean = isFilled
  var isFilled = false
  //empty data matrices for training and validation set
  var X_train: DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_train, params.d)
  var X_validation: DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_validation, params.d)
  var X_test: DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_validation, params.d)
	//empty vectors for the labels of training and validation set
	var z_train : DenseVector[Int] = DenseVector.zeros[Int](params.N_train)
	var z_validation : DenseVector[Int] =  DenseVector.zeros[Int](params.N_validation)
  var z_test : DenseVector[Int] =  DenseVector.zeros[Int](params.N_test)

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Synthetic dataset with "+params.d+" variables.\n")
		sb.append("Observations: "+ params.N_train +" (training), " + params.N_validation+ "(validation)"
      + params.N_test+ "(test)\n")
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
    val NtrainAndVal = params.N_train + params.N_validation
    z_validation = z(params.N_train until NtrainAndVal)
    z_test = z(NtrainAndVal until params.N)
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
        X_train(i, ::) := DenseVector(x.toArray).t
      //For the validation set:
      }
      else {
        if (i < NtrainAndVal) {
          X_validation(i - params.N_train, ::) := DenseVector(x.toArray).t
        }else {
          //for the test set
          X_test(i - NtrainAndVal, ::) := DenseVector(x.toArray).t
        }
      }
    }
    isFilled = true
  }
}

/**
  * Data that is stored in the local file system as csv files.
  */
class LocalData extends LData{
  //Matrix with 3 rows for mean, variance, and standard deviation and columns for data columns
  var trainSummary = DenseMatrix.zeros[Double](0,0)
  var means = DenseVector.zeros[Double](0)
  var stdev = DenseVector.zeros[Double](0)
  var N : Int = 0
  var N_train : Int = 0
  var N_validation : Int = 0
  var N_test : Int = 0
  var d : Int = 0
  var isFilled = false
  var validationSetIsFilled = false
  var trainingSetIsFilled = false
  var testSetIsFilled = false
  //empty data matrices for training and validation set
  var X_train, X_validation, X_test : DenseMatrix[Double] = DenseMatrix.zeros[Double](1, 1)
  //empty vectors for the labels of training and validation set
  var z_train, z_validation, z_test : DenseVector[Int] = DenseVector.zeros[Int](1)

  //Was the data set correctly initialized?
  override def isDefined : Boolean = isFilled

  override def toString : String = {
    val sb = new StringBuilder
    sb.append("Empirical dataset from local file system with "+ d+" variables.\n")
    sb.append("Observations: "+ N_train +" (training), " + N_validation+ " (validation)\n")
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
    println("mean:\t\tvariance:\t\tstandard deviation:")
    println(trainSummary.t)
    //z-transformation of X_train!
    means = columnMeans(X_train)
    stdev = columnStandardDeviation(X_train)
    val N = X_train.rows
    val Means = tile(means.t, 1, N)
    val SD = tile(stdev.t, 1, N)
    X_train = (X_train - Means) / SD
    trainSummary = summary(Train)
    println("Summary statistics of train data set AFTER z-transformation:")
    println("mean:\tvariance:\tstandard deviation:")
    println(trainSummary.t)
    z_train = labels
    d = X_train.cols
    N_train = X_train.rows
    trainingSetIsFilled = true
    isFilled = validationSetIsFilled && trainingSetIsFilled
  }

  def readValidationDataSet (path: String, separator: Char, columnIndexClass: Int) : Unit = {
    val csvReader : CSVReader = new CSVReader(path, separator, columnIndexClass)
    val (inputs, labels) = csvReader.read()
    X_validation = inputs
    var testSummary = summary(Validation)
    println("Summary statistics of validation data set BEFORE z-transformation with means and standard deviation of the training set:")
    println("mean:\tvariance:\tstandard deviation:")
    println(testSummary.t)
    val N = X_validation.rows
    //val means_test = columnMeans(X_test)
    //val stdev_test = columnStandardDeviation(X_test)
    //val Means = tile(means_test.t, 1, N)
    //val SD = tile(stdev_test.t, 1, N)
    val Means = tile(means.t, 1, N)
    val SD = tile(stdev.t, 1, N)
    X_validation = (X_validation - Means) / SD
    testSummary = summary(Validation)
    println("Summary statistics of validation data set AFTER z-transformation with means and standard deviation of the training set:")
    println("mean:\tvariance:\tstandard deviation:")
    println(testSummary.t)
    z_validation = labels
    d = X_validation.cols
    N_validation = X_validation.rows
    validationSetIsFilled = true
    isFilled = validationSetIsFilled && trainingSetIsFilled
  }

  def readTestDataSet (path: String, separator: Char, columnIndexClass: Int) : Unit = {
    val csvReader : CSVReader = new CSVReader(path, separator, columnIndexClass)
    val (inputs, labels) = csvReader.read()
    X_test = inputs
    var testSummary = summary(Test)
    println("Summary statistics of test data set BEFORE z-transformation with means and standard deviation of the training set:")
    println("mean:\tvariance:\tstandard deviation:")
    println(testSummary.t)
    val N = X_test.rows
    val Means = tile(means.t, 1, N)
    val SD = tile(stdev.t, 1, N)
    X_test = (X_test - Means) / SD
    testSummary = summary(Test)
    println("Summary statistics of validation data set AFTER z-transformation with means and standard deviation of the training set:")
    println("mean:\tvariance:\tstandard deviation:")
    println(testSummary.t)
    z_test = labels
    d = X_test.cols
    N_test = X_test.rows
    testSetIsFilled = true
    isFilled = validationSetIsFilled && trainingSetIsFilled && testSetIsFilled
  }

  def summary(dataSetType: DataSetType.Value):DenseMatrix[Double]={
    dataSetType match{
      case Validation => createSummary(X_validation)
      case Train => createSummary(X_train)
      case Test => createSummary(X_test)
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  def columnMeans(m: DenseMatrix[Double]):DenseVector[Double] = mean(m(::,*)).t
  def columnVariance(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t
  def columnStandardDeviation(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t.map(x=>sqrt(x))

  def createSummary(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val summaryM = DenseMatrix.zeros[Double](3,m.cols)
    summaryM.t(::,0) := columnMeans(m)
    summaryM.t(::,1) := columnVariance(m)
    summaryM.t(::,2) := columnStandardDeviation(m)
    summaryM
  }
  override def getN_Train: Int = N_train
  override def getN_Validation: Int = N_validation
  override def getN_Test: Int = N_test
  override def getd: Int = d
}
