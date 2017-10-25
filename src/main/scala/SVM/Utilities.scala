package SVM
import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps 
import breeze.linalg.operators 
import breeze.stats.distributions._

class Data(val par: DataParams){
	
	//empty data matrices for training and test set
        var X_train = DenseMatrix.zeros[Double](par.N_train, par.d)
        var X_test = DenseMatrix.zeros[Double](par.N_test, par.d)

	//empty vectors for the labels of training and test set
	var z_train = DenseVector.zeros[Double](par.N_train)
        var z_test =  DenseVector.zeros[Double](par.N_test)

	def getN_train() : Int = par.N_train
	def getN_test()  : Int = par.N_test
	def getd()  : Int = par.d

def simulate() : Unit = {

	// Set locations of two modes, theta1 and theta2
	val theta1 = DenseVector.rand(par.d)
	val theta2 = DenseVector.rand(par.d)

	//Randomly allocate observations to each class (0 or 1)
	val z = DenseVector.rand(par.N).map(x=>2*x).map(x=>floor(x)).map(x=>(2*x)-1)
	z_train = z(0 until par.N_train)
	z_test = z(par.N_train until par.N)

	val mvn1 = breeze.stats.distributions.MultivariateGaussian(theta1, diag(DenseVector.fill(par.d){1.0}))
	val mvn2 = breeze.stats.distributions.MultivariateGaussian(theta2, diag(DenseVector.fill(par.d){1.0}))

	// Simulate each observation depending on the component its been allocated
	// Create all inputs (predictors)
	var x = DenseVector.zeros[Double](par.d) 

	for(i <- 0 to (par.N - 1)){
  		if ( z(i) == 0 ) {
    			x = mvn1.sample()
  		}else{
    			x = mvn2.sample()
  		}
  
		//Matrix assignment to row
		if( i < par.N_train ){
			X_train(i, ::) := DenseVector(x.toArray).t
		}else{
			X_test(i - par.N_train, ::) := DenseVector(x.toArray).t
		}
	}
}

}
