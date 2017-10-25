package SVM

abstract class Parameters

/**
 * sigma: Variance of the RBF kernel 
 * epsilon: Threshold for similarity between data instances (If k(x,y) < epsilon then we approximate with 0!)
 * C: The parameter C of the C-SVM model.
 * lambda: The regularization parameter.
 * batchProb: Probability for being part of the active set (determines batch size).
 */
case class ModelParams(val sigma: Double, val epsilon: Double, 
    val C: Double, val lambda: Double, val batchProb: Double) extends Parameters {
 
  assert(sigma>0.0)
  assert(epsilon>0.0)
  assert(C>0)
  assert(lambda>=0.0)
  assert(batchProb>0.0 && batchProb<1.0)
  
  /**
   * Define a learning rate delta. 
   * Note that delta has to be smaller than 1/lambda for the algorithm to work!
   */
  var delta = 0.1/lambda  
}

  /**
 * N: Number of observations in the artificial test set.
 * d: The number of features (i.e. inputs, variables).
 * ratioTrain: The ratio of data used for the training set.
  */
case class DataParams(val N: Int, val d : Int, val ratioTrain: Double) extends Parameters{

  assert(N>2)
  assert(d>1)
  assert(ratioTrain>0.0 && ratioTrain<1.0)
 
  /**
   * Number of observations in the training set.  
   */
  val N_train = Math.floor(N * ratioTrain).toInt
  
   /**
   * Number of observations in the test set.  
   */
  val N_test = Math.floor(N * (1.0 - ratioTrain)).toInt
 
  assert(N == N_train + N_test)
}

