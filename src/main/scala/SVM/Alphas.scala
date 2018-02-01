package SVM

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{abs, pow, sqrt}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister

import scala.math.{max, min}

object Alphas{
  /**
    * Setting the momentum to zero when <0 is equivalent to resetting the algorithm because we "forget" previous search directions:
    *
    * "The Fletcher-Reeves method converges if the starting point is sufficiently close to the desired minimum,
      whereas the Polak-Ribiere method can, in rare cases, cycle infinitely without converging. However, Polak-
      Ribiere often converges much more quickly.
      Fortunately, convergence of the Polak-Ribiere method can be guaranteed by choosing max{beta,0}.
      Using this value is equivalent to restarting CG if beta < 0.
      To restart CG is to forget past search directions,
      and start CG anew in the direction of steepest descent."

    Source: Page 42, 14.1. Outline of the Nonlinear Conjugate Gradient Method, in:
    "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
    Jonathan Richard Shewchuk, August 4, 1994
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    */
  val minMomentum : Double = 0.00
  def mean (d: Double, d1: Double): Double = 0.5 * (d + d1)
}

/**
  * N: The number of observations in the training set
  **/
case class Alphas(N: Int,
                  alpha: DenseVector[Double],
                  alphaOld: DenseVector[Double]){

  /* Here I have to initiate an implicit seed generator.
   There are two scenarios:
   In scenario 1, the alphas are initiated in only one thread.
    In this case it would be sufficient to create the following implicit seed:
     implicit val basis: RandBasis = RandBasis.withSeed(0)
   In scenario 2, alphas are initiated in several threads. In this case the different threads should not work with the same seed.
   The necessary implicit seed can be initiated like this:
   val int = new AtomicInteger(1234567)
   new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(int.getAndIncrement())))

   Quote from David Hall (main Breeze contributor, compare https://github.com/scalanlp/breeze/pull/493):
    "In Breeze, the RandBasis is passed around as an implicit argument.
    implicit val basis: RandBasis = RandBasis.withSeed(0)
    val myGenerator = new Gaussian(0, 1) // (basis) is implicitly passed in rather than the default argument.
    (The scoping rules for implicits are somewhat complex, but basically anything in lexical scope is available.)
    Because they can be used this way, it's possible for the same basis to end up on two different threads. With your implementation, both threads will be given their own generator with the same initial seed, meaning they will generate precisely the same sequence of random variables. This is almost certainly bad.
      If, instead, we use the atomic int thing, each thread is seeded with a unique random seed. Because the first seed is 0, the first thread that uses it will be initialized to 0, as in the current implementation.
      It's possible we should figure out something that's more stable than that (since there's a data race for who gets which seed), maybe using the name of the thread(?!?), but at least it doesn't produce a catastrophic failure like generating the same variates on every thread.
    So, concretely, I just want the body of the method to be this:
    val int = new AtomicInteger(seed)
    new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(int.getAndIncrement())))*/

  val int = new AtomicInteger(1234567)
  new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(int.getAndIncrement())))

  //Secondary constructor with random default values for the alphas
  def this(N: Int, mp: ModelParams) {
    this(N, mp.C * (DenseVector.ones[Double](N) - DenseVector.rand(N)), mp.C *(DenseVector.ones[Double](N) - DenseVector.rand(N)))
  }

  /**
    *
    * @return
    */
  def getSortedAlphas : Array[Double] = alpha.toArray.sorted[Double]

  def getQuantile (quantile: Double) : Double = {
    assert(quantile>=0 && quantile<=1.0)
    if(quantile == 0.0) return alpha.reduce(min(_,_))  
    if(quantile == 1.0) return alpha.reduce(max(_,_))
    val sortedAlphas : Array[Double] = getSortedAlphas
    val N = alpha.length
    val x = (N+1) * quantile
    val rank_high : Int = min(Math.ceil(x).toInt,N)
    val rank_low : Int = max(Math.floor(x).toInt,1)
    if(rank_high==rank_low) (sortedAlphas(rank_high-1))
    else Alphas.mean(sortedAlphas(rank_high-1), sortedAlphas(rank_low-1))
  }

  /**
    * @param quantile The quantile of empirical distribution where the alphas are cut-off and set to zero.
    * @return The new vector with cutOff % of elements zet to zero.
    */
  def clipAlphas (quantile: Double) : Alphas = {
    assert(quantile>=0 && quantile<=0.999)
    copy(alpha=alpha.map(x => if (x < getQuantile(quantile)) 0 else x))
  }

  /**
    * Returns relative change in alphas.
    * If the norm of the old alphas is smaller epsilon it returns the sum of absolute values of the new alphas.
    * @return
    */
  def getDeltaL1 : Double = sum(abs(alpha - alphaOld))

  def updateAlphaAsConjugateGradient() : Alphas = {
    val diff : DenseVector[Double] = alpha - alphaOld
    val dotProduct : Double = alpha.t * diff
    val alphaOldInnerProduct : Double = alphaOld.t * alphaOld
    if(alphaOldInnerProduct > 0.001){
      val tentativeMomentum = dotProduct / alphaOldInnerProduct
      val momentum : Double = max(tentativeMomentum, Alphas.minMomentum)
      printf("Momentum %.3f ", momentum)
      val alphaUpdated : DenseVector[Double] = alpha + momentum * alphaOld
      //Return a copy of this object with alpha updated according to the
      //Polak-Ribiere conjugate gradient formula.
      //Compare: https://en.wikipedia.org/wiki/Conjugate_gradient_method
      copy(alpha = alphaUpdated, alphaOld = alpha)
    }else{
     this
    }
  }
}
