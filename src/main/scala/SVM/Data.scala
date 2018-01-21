package SVM
import SVM.DataSetType.{Validation, Train}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.sql.{Dataset, SparkSession}
import breeze.stats._

trait Data{

  //Get row with row index (starting with 0) from validation set.
  def getRowValidation (rowIndex: Int):DenseVector[Double]

  //Get row with row index (starting with 0) from training set.
  def getRowTrain(rowIndex: Int):DenseVector[Double]

  //Get label with row index (starting with 0) from validation set.
  def getLabelValidation (rowIndex: Int): Double

  //Get label with row index (starting with 0) from training set.
  def getLabelTrain(rowIndex: Int): Double

  //Get vector of labels from validation set.
  def getLabelsValidation : DenseVector[Int]

  //Get vector of labels from training set.
  def getLabelsTrain : DenseVector[Int]

  //Was the data set correctly initialized?
  def isDefined : Boolean
  def getN_train : Int
  def getN_Validation : Int
  def getd : Int

  //print distribution of labels in validation and training set
  def tableLabels(): Unit
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

class SparkDataSet[T <: basicDataSetEntry](dataSetTrain: Dataset[T], dataSetValidation: Dataset[T]) extends Data{

  def tableLabels(): Unit = {

    val z_train = getLabels(Train)
    val positive_train = z_train.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_train = z_train.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Training: "+positive_train+"(+)/"+negative_train+"(-)/"+z_train.length+"(total)")

    //TODO Add output for test set
    val z_validation = getLabels(Validation)
    val positive_test = z_validation.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_test = z_validation.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Validation: "+positive_test+"(+)/"+negative_test+"(-)/"+z_validation.length +"(total)")
  }

  def getRow(rowIndex: Int, dataSetType: DataSetType.Value): T = {
    if(dataSetType == Validation) return dataSetValidation.filter(x => x.rowNr == rowIndex).first()
    if(dataSetType == Train) dataSetTrain.filter(x => x.rowNr == rowIndex).first()
    else throw new Exception("Unsupported data set type!")
  }

  //Get vector of labels
  def getLabels (dataSetType: DataSetType.Value): DenseVector[Int] = {
    //I have to import implicits here to be able to extract the label from the data set.
    //https://stackoverflow.com/questions/39151189/importing-spark-implicits-in-scala#39173501
    val sparkSession = SparkSession.builder.getOrCreate()
    import sparkSession.implicits._
    if(dataSetType == Validation)return new DenseVector(dataSetValidation.map(x => x.label).collect())
    if(dataSetType == Train)new DenseVector(dataSetTrain.map(x => x.label).collect())
    else throw new Exception("Unsupported data set type!")
 }

  //Get row with row index (starting with 0) from validation set.
  def getRowValidation (rowIndex: Int):DenseVector[Double] = getRow(rowIndex, Validation).getPredictors

  //Get row with row index (starting with 0) from training set.
  def getRowTrain(rowIndex: Int):DenseVector[Double] = getRow(rowIndex, Train).getPredictors

  //Get label with row index (starting with 0) from validation set.
  def getLabelValidation (rowIndex: Int): Double = getRow(rowIndex, Validation).getLabel

  //Get label with row index (starting with 0) from training set.
  def getLabelTrain(rowIndex: Int): Double = getRow(rowIndex, Train).getLabel

  //Get vector of labels from validation set.
  def getLabelsValidation : DenseVector[Int] = {
    getLabels (Validation)
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

  def getN_Validation : Int = {
    dataSetValidation.count().toInt
  }

  def getd : Int = {
    dataSetValidation.first().getD
  }
}

/**
  * Simulated data with given data parameters.
  * @param params
  */
class SimData (val params: DataParams) extends Data {

  def getN_train : Int = params.N_train
  def getN_Validation : Int = params.N_validation
  def getd : Int = params.d

  def tableLabels(): Unit = {

    val positive_train = z_train.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_train = z_train.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Training: "+positive_train+"(+)/"+negative_train+"(-)/"+z_train.length+"(total)")

    val positive_test = z_validation.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_test = z_validation.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Validation: "+positive_test+"(+)/"+negative_test+"(-)/"+z_validation.length +"(total)")
  }

  //Get column with column index (starting with 0) from validation set.
  override def getRowValidation (columnIndex: Int): DenseVector[Double] = {
      X_validation(columnIndex,::).t
  }

  //Get column with column index (starting with 0) from training set.
  override def getRowTrain(columnIndex: Int): DenseVector[Double] = {
      X_train(columnIndex,::).t
  }

  //Get label with row index (starting with 0) from validation set.
  override def getLabelValidation (rowIndex: Int): Double = {
      z_validation(rowIndex)
  }

  //Get label with row index (starting with 0) from training set.
  override def getLabelTrain(rowIndex: Int): Double = {
      z_train(rowIndex)
  }

  //Get vector of labels from validation set.
  override def getLabelsValidation: DenseVector[Int] = {
      z_validation
  }

  //Get vector of labels from training set.
  override def getLabelsTrain: DenseVector[Int] = {
      z_train
  }

  //Was the data set correctly initialized?
  override def isDefined : Boolean = isFilled

  var isFilled = false

  //empty data matrices for training and validation set
  val X_train: DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_train, params.d)
  val X_validation: DenseMatrix[Double] = DenseMatrix.zeros[Double](params.N_validation, params.d)

	//stores the maximum of the square of the euclidean norm
	//var tau : Double = 0.0

	//empty vectors for the labels of training and validation set
	var z_train : DenseVector[Int] = DenseVector.zeros[Int](params.N_train)
	var z_validation : DenseVector[Int] =  DenseVector.zeros[Int](params.N_validation)

	override def toString : String = {
		val sb = new StringBuilder
		sb.append("Synthetic dataset with "+params.d+" variables.\n")
		sb.append("Observations: "+ params.N_train +" (training), " + params.N_validation+ "(validation)\n")
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
      z_validation = z(params.N_train until params.N)

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
          X_validation(i - params.N_train, ::) := DenseVector(x.toArray).t
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
  var trainSummary = DenseMatrix.zeros[Double](0,0)
  var means = DenseVector.zeros[Double](0)
  var stdev = DenseVector.zeros[Double](0)


  //Get column with column index (starting with 0) from validation set.
    override def getRowValidation (columnIndex: Int): DenseVector[Double] = {
      X_validation(columnIndex,::).t
    }

    //Get column with column index (starting with 0) from training set.
    override def getRowTrain(columnIndex: Int): DenseVector[Double] = {
      X_train(columnIndex,::).t
    }

    //Get label with row index (starting with 0) from validation set.
    override def getLabelValidation (rowIndex: Int): Double = {
      z_validation(rowIndex)
    }

    //Get label with row index (starting with 0) from training set.
    override def getLabelTrain(rowIndex: Int): Double = {
      z_train(rowIndex)
    }

    //Get vector of labels from validation set.
    override def getLabelsValidation: DenseVector[Int] = {
      z_validation
    }

    //Get vector of labels from training set.
    override def getLabelsTrain: DenseVector[Int] = {
      z_train
    }

    //Was the data set correctly initialized?
    override def isDefined : Boolean = isFilled

    var N : Int = 0
    var N_train : Int = 0
    var N_validation : Int = 0
    var d : Int = 0

    var isFilled = false
    var validationSetIsFilled = false
    var trainingSetIsFilled = false

    //empty data matrices for training and validation set
    var X_train, X_validation : DenseMatrix[Double] = DenseMatrix.zeros[Double](1, 1)

    //empty vectors for the labels of training and validation set
    var z_train, z_validation : DenseVector[Int] = DenseVector.zeros[Int](1)

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

  /**
    * @param dataSetType
    * @return
    */
  def summary(dataSetType: DataSetType.Value):DenseMatrix[Double]={
    if(dataSetType == Validation) return createSummary(X_validation)
    if(dataSetType == Train) createSummary(X_train)
    else throw new Exception("Unsupported data set type!")
  }

  def columnMeans(m: DenseMatrix[Double]):DenseVector[Double] = mean(m(::,*)).t
  def columnVariance(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t
  def columnStandardDeviation(m: DenseMatrix[Double]):DenseVector[Double] = variance(m(::,*)).t.map(x=>sqrt(x))


  def tableLabels(): Unit = {

    val positive_train = z_train.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_train = z_train.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Training: "+positive_train+"(+)/"+negative_train+"(-)/"+z_train.length+"(total)")

    val positive_test = z_validation.map(x => if(x>0)1 else 0).reduce(_+_)
    val negative_test = z_validation.map(x => if(x<0)1 else 0).reduce(_+_)
    println("Validation: "+positive_test+"(+)/"+negative_test+"(-)/"+z_validation.length +"(total)")
  }

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

  override def getN_Validation: Int = N_validation

  override def getd: Int = d
}
