package SVM
import breeze.linalg._
import breeze.numerics._

abstract class Data(params: DataParams){

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

  def getN_train : Int = params.N_train
  def getN_test : Int = params.N_test
  def getd : Int = params.d
}

class simData(val params: DataParams) extends Data(params) {

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
	var tau : Double = 0.0

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
      val z = DenseVector.rand(params.N).map(x=>2*x).map(x=>floor(x)).map(x=>(2*x)-1)
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
          val norm = sqrt(x.map(e => abs(e)).reduce(_+_))
          //update tau as maximum over all norms
          tau = max(tau, norm)
          X_train(i, ::) := DenseVector(x.toArray).t
        //For the tests set:
        }else{
          X_test(i - params.N_train, ::) := DenseVector(x.toArray).t
        }
      }
      isFilled = true
    }
  }


