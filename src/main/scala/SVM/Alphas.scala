package SVM

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{abs, pow, sqrt}

/**
  * N: The number of observations in the training set
  **/
case class Alphas(N: Int,
                  alpha: DenseVector[Double],
                  alphaOld: DenseVector[Double]){

  //Secondary constructor with random default values for the alphas
  def this(N: Int) {
    this(N, DenseVector.ones[Double](N) - DenseVector.rand(N), DenseVector.ones[Double](N) - DenseVector.rand(N))
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
