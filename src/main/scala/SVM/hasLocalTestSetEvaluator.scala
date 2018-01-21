package SVM

import breeze.linalg.{DenseVector, min}
import breeze.numerics.signum
import SVM.DataSetType.{Test, Train, Validation}

trait hasLocalTestSetEvaluator extends Algorithm{

  /**
    * Returns the predicted class (-1 or +1) for a test set.
    *
    * alphas: The alpha parameters.
    * ap:     AlgoParams object storing parameters of the algorithm
    * kmf:    LocalKernelMatrixFactory that contains the local matrices for the data set
    ***/
  def evaluateOnTestSet(alphas: Alphas, ap: AlgoParams, kmf: LocalKernelMatrixFactory):DenseVector[Double]= {

    //Get the distributed kernel matrix for the test set:
    val S = kmf.S
    assert(S.cols>0, "The number of columns of S is zero.")
    assert(S.rows>0, "The number of rows of S is zero.")

    val A : DenseVector[Double] = alphas.alpha *:* kmf.getData().getLabels(Train).map(x=>x.toDouble)
    assert(A!=null && S!=null, "One of the input matrices is undefined!")
    assert(A.length>0, "The number of elements of A is zero.")
    assert(A.length==S.rows,"The number of elements of A does not equal the number of rows of S!")

    val maxPrintIndex = min(alphas.alpha.length, 10)
    if (ap.isDebug) {
      println("alphas:" + alphas.alpha(0 until maxPrintIndex))
      println("z_train:" + kmf.getData().getLabels(Train)(0 until maxPrintIndex))
      println("A:" + A(0 until maxPrintIndex))
    }
    val P = A.t * S
    if (ap.isDebug) {
      println("P:" + P.t(0 until maxPrintIndex))
    }
    //Return the predictions
    signum(P).t
  }
}