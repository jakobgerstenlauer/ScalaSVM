package SVM

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{abs, pow, sqrt}
import scala.math.{min,max}

/**
  * N: The number of observations in the training set
  **/
case class Alphas(N: Int,
                  alpha: DenseVector[Double],
                  alphaOld: DenseVector[Double]){

  //Secondary constructor with random default values for the alphas
  def this(N: Int, mp: ModelParams) {
    this(N, mp.C * (DenseVector.ones[Double](N) - DenseVector.rand(N)), mp.C *(DenseVector.ones[Double](N) - DenseVector.rand(N)))
  }

  private def getSortedAlphas : Array[Double] = alpha.toArray.sorted[Double]

  def mean (d: Double, d1: Double): Double = 0.5 * (d + d1)

  def getQuantile (quantile: Double) : Double = {
    if(quantile == 0.0) return alpha.reduce(min(_,_))  
    if(quantile == 1.0) return alpha.reduce(max(_,_))
    val sortedAlphas : Array[Double] = getSortedAlphas
    val N = alpha.length
    val x = (N+1) * quantile
    val rank_high : Int = min(Math.ceil(x).toInt,N)
    val rank_low : Int = max(Math.floor(x).toInt,1)
    if(rank_high==rank_low) return (sortedAlphas(rank_high-1))
    else return mean(sortedAlphas(rank_high-1), sortedAlphas(rank_low-1))
  }

  /**
    * @param quantile The quantile of empirical distribution where the alphas are cut-off and set to zero.
    * @return The new vector with cutOff % of elements zet to zero.
    */
  def clipAlphas (quantile: Double) : Alphas = {
    assert(quantile>=0 && quantile<=0.999)
    copy(alpha=alpha.map(x => if (x < getQuantile(quantile)) 0 else x))
  }

  def getDelta : Double = sum(abs(alpha - alphaOld))

  def updateAlphaAsConjugateGradient() : Alphas = {
    val diff : DenseVector[Double] = alpha - alphaOld
    val dotProduct : Double = alpha.t * diff
    val alphaOldNorm : Double = sqrt(alphaOld.map(x => pow(x,2)).reduce(_ + _))
    if(alphaOldNorm > 0.000001){
      val momentum : Double = dotProduct / alphaOldNorm
      printf("Momentum %.3f ", momentum)
      val alphaUpdated : DenseVector[Double] = alpha + momentum * alphaOld
      val alphaUpdatedNorm : Double = sqrt(alphaUpdated.map(x => pow(x,2)).reduce(_ + _))
      //Return a copy of this object with alpha updated according to the
      //Polak-Ribiere conjugate gradient formula.
      //Compare: https://en.wikipedia.org/wiki/Conjugate_gradient_method
      copy(alpha = alphaUpdated / alphaUpdatedNorm, alphaOld = alpha)
    }else{
      val momentum = 0.01
      printf("Momentum %.3f ", momentum)
      val alphaUpdated : DenseVector[Double] = alpha + momentum * alphaOld
      val alphaUpdatedNorm : Double = sqrt(alphaUpdated.map(x => pow(x,2)).reduce(_ + _))
      //If the norm of alpha in the previous step is below a threshold,
      //return a copy of this object without any changes.
      copy(alpha = alphaUpdated / alphaUpdatedNorm, alphaOld = alpha)
    }
  }
}
