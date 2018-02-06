package SVM
abstract class Parameters

/**
  * Properties of algorithms
  * @param maxIter The maximum number of iterations of the algorithm.
  * @param minDeltaAlpha The threshold of changes in absolute sum over all elements of alpha below which the algorithm stops.
  * @param learningRateDecline The rate of decline of the learning rate from iteration to iteration.
  * @param batchProb Probability for being part of the active set (determines batch size).
  * @param epsilon Threshold for similarity between data instances (If k(x,y) < epsilon then we approximate with 0!)
  * @param isDebug Should the algorithm be verbose?
  * @param quantileAlphaClipping Setting sparsity lower quantile of alphas to zero in each iteration. Can be tuned to enforce stronger sparsity.
  */
case class AlgoParams(maxIter: Int = 30, minDeltaAlpha: Double = 0.01, learningRateDecline: Double = 0.5, batchProb: Double = 0.8, epsilon : Double = 0.0001, isDebug: Boolean = false, quantileAlphaClipping: Double = 0.0) extends Parameters{
  assert(quantileAlphaClipping>=0 && quantileAlphaClipping<=0.99)
  assert(batchProb>0.0 && batchProb<1.0)
  assert(learningRateDecline <= 1.0 && learningRateDecline > 0.0)
  assert(epsilon >= 0)
}

/**
  * Model parameters
  * @param C The parameter C of the C-SVM model.
  * @param delta The learning rate or step size.
  */
case class ModelParams(C: Double = 1.0, delta: Double = 0.5) extends Parameters {
 
  assert(C>0, "C must be positive!")

  def updateDelta(ap: AlgoParams): ModelParams = {
	  copy(delta = delta * ap.learningRateDecline)
  }

  override def toString : String = {
    val sb = new StringBuilder
    sb.append("Model parameters: \n")
    sb.append("C: "+C+"\n")
    sb.append("delta: "+delta+"\n")
    sb.toString()
  }  
}

  /**
 * N: Number of observations in the artificial data set.
 * d: The number of features (i.e. inputs, variables).
 * ratioTrain: The ratio of data used for the training set.
  */
case class DataParams(N: Int = 1000, d: Int = 5, ratioTrain: Double = 0.5, ratioTest: Double = 0.1) extends Parameters{

  assert(N>2)
  assert(d>1)
  assert(ratioTrain>0.0 && ratioTrain<1.0)
  assert(N*ratioTest%4==0,"Nr of instances in test set must be a multiple of four!")
  assert(N*(1-ratioTest-ratioTrain)%4==0,"Nr of instances in validation set must be a multiple of four!")


    /**
   * Number of observations in the training set.  
   */
  val N_train = Math.floor(N * ratioTrain).toInt
  
   /**
   * Number of observations in the validation set.
   */
  val N_validation = Math.floor(N * (1.0 - (ratioTrain+ratioTest))).toInt

  /**
    * Number of observations in the validation set.
    */
  val N_test = Math.floor(N * ratioTest).toInt

  assert(N == N_train + N_validation + N_test)

  override def toString : String = {
        val sb = new StringBuilder
        sb.append("Data parameters: \n")
        sb.append("Total number of observations: "+N+"\n")
        sb.append("Observations training set: " + N_train + "\n")
        sb.append("Observations validation set: " + N_validation + "\n")
        sb.append("Observations test set: " + N_test + "\n")
        sb.append("Number of features: " + d + "\n")
        sb.toString()
  }
}

