package SVM
import breeze.linalg.{DenseVector, _}
import breeze.numerics._

trait hasLocalTrainingSetEvaluator extends Algorithm{

	/**
	* Returns the predicted class (-1 or +1) for a test set.
	*
	* alphas: The alpha parameters.
	* ap:     AlgoParams object storing parameters of the algorithm
	* kmf:    LocalKernelMatrixFactory that contains the local matrices for the data set
	***/
	def evaluateOnTrainingSet(alphas: Alphas, ap: AlgoParams, kmf: LocalKernelMatrixFactory):DenseVector[Double]= {
		//Get the kernel matrix for the training set:
		val K = kmf.K
    val z = kmf.z.map(x=>x.toDouble)
    assert(K.cols>0, "The number of columns of K is zero.")
    assert(K.rows>0, "The number of rows of K is zero.")

		val A : DenseVector[Double] = alphas.alpha *:* z
    assert(A!=null && K!=null, "One of the input matrices is undefined!")
    assert(A.length>0, "The number of elements of A is zero.")
    assert(A.length==K.rows,"The number of elements of A does not equal the number of rows of S!")

    val maxPrintIndex = min(alphas.alpha.length, 10)
    if (ap.isDebug) {
      println("alphas:" + alphas.alpha(0 until maxPrintIndex))
      println("z:" + z(0 until maxPrintIndex))
      println("A:" + A(0 until maxPrintIndex))
    }
		val P: Transpose[DenseVector[Double]] = A.t * K
    if (ap.isDebug) {
      println("P:" + P.t(0 until maxPrintIndex))
    }
    //Return the predictions
    signum(P).t
	}
}
