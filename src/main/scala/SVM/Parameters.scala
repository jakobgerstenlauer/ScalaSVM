package SVM

abstract class Parameters

/**
* maxIter: The maximum number of iterations of the algorithm.
* minDeltaAlpha: The threshold of changes in absolute sum over all elements of alpha below which the algorithm stops.
* learningRateDecline: The rate of decline of the learning rate from iteration to iteration.
* numBaggingReplicates: The number of bagging replicates (number of different randomly selected observations used for estimating the new alpha).
* batchProb: Probability for being part of the active set (determines batch size).
* epsilon: Threshold for similarity between data instances (If k(x,y) < epsilon then we approximate with 0!)
* isDebug: Should the algorithm be verbose? 
**/
case class AlgoParams(val maxIter: Int = 30, val minDeltaAlpha: Double = 0.001, val learningRateDecline: Double = 0.95, val numBaggingReplicates: Int = 100, val batchProb: Double = 0.1, val epsilon : Double = 0.0001, val isDebug: Boolean = false) extends Parameters{
  assert(batchProb>0.0 && batchProb<1.0)
  assert(learningRateDecline <= 1.0 && learningRateDecline > 0.0)
  assert(epsilon >= 0)
  assert(numBaggingReplicates > 0)
}

/**
 * C: The parameter C of the C-SVM model.
 * lambda: The regularization parameter.
 */
case class ModelParams(val C: Double = 1.0, val lambda: Double = 0.1, var delta: Double = 0.5) extends Parameters {
 
  assert(C>0)
  assert(lambda>=0.0)
  
  /**
   * Define a VARIABLE learning rate delta. 
   * Note that delta has to be smaller than 1/lambda for the algorithm to work!
   */
  if(delta >= 1/lambda) delta = 0.99 * (1/lambda) 

  def updateDelta(learningRateDecline: Double):Unit={
	delta = delta * learningRateDecline
  }

  override def toString : String = {
        val sb = new StringBuilder
        sb.append("Model parameters: \n")
        sb.append("C: "+C+"\n")
        sb.append("lambda: "+lambda+"\n")
        return sb.toString()
  }  
}

  /**
 * N: Number of observations in the artificial test set.
 * d: The number of features (i.e. inputs, variables).
 * ratioTrain: The ratio of data used for the training set.
  */
case class DataParams(val N: Int = 1000, val d : Int = 5, val ratioTrain: Double = 0.5) extends Parameters{

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

  override def toString : String = {
        val sb = new StringBuilder
        sb.append("Data parameters: \n")
        sb.append("Total number of observations: "+N+"\n")
        sb.append("Observations training set: " + N_train + "\n")
        sb.append("Observations test set: " + N_test + "\n")
        sb.append("Number of features: " + d + "\n")
        return sb.toString()
  }  

}

