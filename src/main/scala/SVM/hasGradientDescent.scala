package SVM
import breeze.linalg.{DenseVector, _}
import breeze.stats.distributions._
import breeze.stats.DescriptiveStats._
import breeze.numerics._

trait hasGradientDescent extends Algorithm{

  /**
    *
    * @param alphas The input vector of alphas.
    * @param cutOff The quantile of the log-normal distribution where the alphas are cut-off and set to zero.
    * @return The new vector with cutOff % of elements zet to zero.
    */
  def clipAlphas (alphas: DenseVector[Double], cutOff: Double) : DenseVector[Double] = {
    val meanAndVar = breeze.stats.meanAndVariance (alphas.map(x => log (x) ) )
    val logNormalDist = new LogNormal (meanAndVar.mean, meanAndVar.variance)
    //set the lower cutoff% quantile of alphas to zero:
    alphas.map (x => if (logNormalDist.cdf (x) < 0.1) 0 else x)
  }

  def calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): DenseVector[Double] = {
    //Extract model parameters
    val N = alphas.alpha.length
    val maxPrintIndex = min(10, N)
    val C = mp.C
    val d = kmf.getData().getLabelsTrain.map(x => x.toDouble)
    val delta = mp.delta
    val shrinking = ap.learningRateDecline
    val shrinkedValues: DenseVector[Double] = shrinking * alphas.alpha
    val gradient = kmf.calculateGradient(alphas.alpha)

    if (ap.isDebug) {
      println("gradient: " + gradient(0 until maxPrintIndex))
      println("alphas before update:" + alphas.alpha(0 until maxPrintIndex))
    }

    //Our first, tentative, estimate of the updated parameters is:
    val alpha1: DenseVector[Double] = alphas.alpha - delta *:* gradient
    if (ap.isDebug) {
      println("alphas first tentative update:" + alpha1(0 until maxPrintIndex))
    }

    //Then, we have to project the alphas onto the feasible region defined by the first constraint:
    val alpha2: DenseVector[Double] = alpha1 - (d * (d dot alpha1)) / (d dot d)
    if (ap.isDebug) {
      println("alphas after projection:" + alpha2(0 until maxPrintIndex))
    }

    //The value of alpha has to be between 0 and C.
    val updated = alpha2.map(alpha => if (alpha > C) C else alpha).map(alpha => if (alpha > 0) alpha else 0.0)
    if (ap.isDebug) println("alphas after projection:" + updated(0 until maxPrintIndex))

    //random boolean vector: is a given observation part of the batch? (0:no, 1:yes)
    val batchProb: Double = ap.batchProb
    val randomUniformNumbers: DenseVector[Double] = DenseVector.rand(N)
    val isInBatch: DenseVector[Double] = randomUniformNumbers.map(x => if (x < batchProb) 1.0 else 0.0)
    if (ap.isDebug) println("batch vector: " + isInBatch(0 until maxPrintIndex))

    val tuples = (isInBatch.toArray.toList zip shrinkedValues.toArray.toList zip updated.toArray.toList) map { case ((a, b), c) => (a, b, c) }
    val stochasticUpdate = new DenseVector(tuples.map { case (inBatch, alphas_shrinked, alphas_updated) => if (inBatch == 1) alphas_updated else alphas_shrinked }.toArray)
    if (ap.isDebug) println("stochastic update:" + stochasticUpdate(0 until maxPrintIndex))

    clipAlphas(stochasticUpdate, ap.sparsity)
  }
}
