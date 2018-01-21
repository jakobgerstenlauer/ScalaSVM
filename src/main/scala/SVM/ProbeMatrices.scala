package SVM

import SVM.DataSetType.{Validation, Train}
import breeze.linalg.max

case class ProbeMatrices(d: Data, kf: KernelFunction){

  /**
    * Returns the number of non-sparse matrix elements for a given epsilon
    * @param typeOfMatrix For the training matrix K or the test matrix S?
    * @param epsilon Value below which similarities will be ignored.
    * @return
    */
  def probeSparsity(typeOfMatrix: DataSetType.Value, epsilon: Double): Int = {
    typeOfMatrix match{
      case Train => probeSparsityK(epsilon)
      case Validation => probeSparsityS(epsilon)
      case _ => throw new Exception("Unsupported data set type!")
    }
  }

  /**
    * Returns the number of non-sparse matrix elements for a given epsilon and the training matrix K.
    * @param epsilon Value below which similarities will be ignored.
    * @return
    */
  def probeSparsityK(epsilon: Double): Int = {
    val N = d.getN_Train
    var size2 : Int = N //the diagonal elements are non-sparse!
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N; j <- (i+1) until N if(kf.kernel(d.getRowTrain(i), d.getRowTrain(j)) > epsilon)){
      size2 = size2 + 2
    }
    size2
  }

  /**
    * Returns the number of non-sparse matrix elements for a given epsilon and the training vs test matrix S.
    * @param epsilon
    * @return
    */
  def probeSparsityS(epsilon: Double): Int = {
    var size2 : Int = 0
    val N_train = d.getN_Train
    val N_test = d.getN_Validation
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N_test; j <- (i+1) until N_train
         if(kf.kernel(d.getRowValidation(i), d.getRowTrain(j)) > epsilon)){
      size2 = size2 + 2
    }
    //iterate over the diagonal
    for (i <- 0 until max(N_test,N_train)
         if(kf.kernel(d.getRowValidation(i), d.getRowTrain(i)) > epsilon)){
      size2 = size2 + 1
    }
    size2
  }
}
