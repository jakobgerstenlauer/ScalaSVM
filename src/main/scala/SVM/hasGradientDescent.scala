package SVM
import breeze.linalg.{DenseVector, _}
import SVM.DataSetType.{Test, Train, Validation}

trait hasGradientDescent extends Algorithm {

  def calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): DenseVector[Double] = {
    //Extract model parameters
    val N = alphas.alpha.length
    val maxPrintIndex = min(10, N)
    val C = mp.C
    val d = kmf.getData.getLabels(Train).map(x => x.toDouble)
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
    stochasticUpdate
  }
}
