package SVM

import breeze.linalg.{DenseVector, _}
import org.apache.spark.SparkContext
import SVM.DataSetType.{Train, Validation}
import scala.collection.mutable.{ListBuffer}
import scala.concurrent.{Future, Promise}
import scala.util.{Success,Failure}
case class AllMatrixElementsZeroException(message:String) extends Exception(message)
case class EmptyRowException(message:String) extends Exception(message)
import scala.concurrent.ExecutionContext.Implicits.global

abstract class Algorithm(alphas: Alphas){

  /**
    * Performs one iteration of the algorithm and returns the updated algorithm.
    * @return updated algorithm object.
    */
  def iterate : Algorithm

  /**
    * Calculates the number of correctly and incorrectly classified instances as tuple.
    * @param predictions The vector of class predictions.
    * @param labels The vector of empirical classes.
    * @return A tuple consisting of first the number of correct predictions (classifications)
    *         and second the number of misclassifications.
    */
  def calculateAccuracy(predictions: DenseVector[Double], labels: DenseVector[Int]):(Int,Int) = {
    assert(predictions.length == labels.length)
    val product : DenseVector[Double] = predictions *:* labels.map(x => x.toDouble)
    val correct = product.map(x=>if(x>0) 1 else 0).reduce(_+_)
    val misclassified : Int = predictions.length - correct
    (correct, misclassified)
  }

  private def printAccuracy(correct: Int, misclassified: Int):String = {
    val total = correct + misclassified
    "%d/%d=%d%%".format(correct, total, Math.round(100.0 * (correct.toDouble / total.toDouble)))
  }

  private def printSparsity() : String = "%d%%".format(Math.round(getSparsity))

  def createLog(correct: Int, misclassified : Int, correctT : Int, misclassifiedT : Int, alphas : Alphas):String={
    "Train:"+printAccuracy(correct,misclassified)+
      ",Validation:"+printAccuracy(correctT,misclassifiedT)+
      ",Sparsity:"+printSparsity()
  }

  /**
    * Get the sparsity of the algorithm.
    * @return Sparsity between 0 (0%) and 100 (100%)
    */
  def getSparsity: Double = 100.0 * (alphas.alpha.map(x => if (x == 0) 1 else 0).reduce(_ + _).toDouble / alphas.alpha.length.toDouble)
}

/**
  * Lean implementation of the sequential gradient descent algorithm without matrices.
  * @param alphas The current and old values of the alphas.
  * @param ap Properties of the algorithm
  * @param mp Properties of the model
  * @param kmf A KernelMatrixFactory for local matrices.N
  */
case class NoMatrices(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LeanMatrixFactory, optimalSparsityFuture : ListBuffer[Future[Int]]) extends Algorithm(alphas) with hasGradientDescent {

  def predictOnTestSet() : Future[Int] = {

    val promise = Promise[Int]

    //Turn the ListBuffer into a List
    val listOfFutures : List[Future[Int]] = optimalSparsityFuture.toList

    //List[Future[Int]] => Future[List[Int]]
    val futureList : Future[List[Int]] = Future.sequence(listOfFutures)

    //Wait for cross-validation results to choose the optimal level of sparsity:
    futureList onComplete {
      case Success(res) => {
        assert(res.size>0)
        val sum : Int = res.sum
        val count = res.size
        val optSparsity = 0.01 *(sum.toDouble/count.toDouble)
        println("Based on cross-validation, the optimal sparsity is: "+optSparsity)
        alphas.clipAlphas(optSparsity)
        println("Predict on the test set.")
        val promisedTestResults : Future[Int] = kmf.predictOnTestSet(alphas)
        promise.completeWith(promisedTestResults)
      }
      case Failure(ex) => println(ex)
    }
    promise.future
  }

  def iterate: NoMatrices = {
    if(alphas.getDeltaL1 < ap.minDeltaAlpha)return this
    assert(getSparsity < 99.0)
    //Decrease the step size, i.e. learning rate:
    val ump = mp.updateDelta(ap)
    //Update the alphas using gradient descent
    val algo = gradientDescent(alphas, ap, ump, kmf)
    optimalSparsityFuture.append(kmf.predictOnValidationSet(algo.alphas))
    algo.copy(optimalSparsityFuture=optimalSparsityFuture)
  }

  /**
    * Performs a stochastic gradient update with clipping of the alphas.
    * Note the difference in the return type.
    * @param alphas The primal variables.
    * @param ap The properties of the algorithm.
    * @param mp The properties of the model.
    * @param kmf A MatrixFactory object.
    * @return An updated instance of the algorithm.
    */
  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LeanMatrixFactory): NoMatrices = {
    val stochasticUpdate = calculateGradientDescent (alphas, ap, mp, kmf)
    copy(alphas = alphas.copy(alpha = stochasticUpdate).updateAlphaAsConjugateGradient())
  }
}

/**
  * Sequential gradient descent algorithm with local matrices
  * @param alphas The current and old values of the alphas.
  * @param ap Properties of the algorithm
  * @param mp Properties of the model
  * @param kmf A KernelMatrixFactory for local matrices.
  */
case class SGLocal(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LocalKernelMatrixFactory) extends Algorithm(alphas)
  with hasLocalTrainingSetEvaluator with hasLocalTestSetEvaluator with hasGradientDescent {

  def iterate: SGLocal = {
    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf), kmf.getData().getLabels(Train))
    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf), kmf.getData().getLabels(Validation))
    println(createLog(correct, misclassified, correctT, misclassifiedT, alphas))
    assert(getSparsity < 99.0)
    //Decrease the step size, i.e. learning rate:
    val ump = mp.updateDelta(ap)
    //Update the alphas using gradient descent
    gradientDescent(alphas, ap, ump, kmf)
  }

  /**
    * Performs a stochastic gradient update with clipping of the alphas.
    * Note the difference in the return type.
    * @param alphas The primal variables.
    * @param ap The properties of the algorithm.
    * @param mp The properties of the model.
    * @param kmf A MatrixFactory object.
    * @return An updated instance of the algorithm.
    */
  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): SGLocal = {
    val stochasticUpdate = calculateGradientDescent (alphas, ap, mp, kmf)
    copy(alphas = alphas.copy(alpha = stochasticUpdate).updateAlphaAsConjugateGradient().clipAlphas(ap.quantileAlphaClipping))
  }
}

/**
  * Sequential gradient descent algorithm with distributed matrices
  * @param alphas The current and old values of the alphas.
  * @param ap Properties of the algorithm
  * @param mp Properties of the model
  * @param kmf A KernelMatrixFactory for distributed matrices.
  * @param sc The Spark context of the cluster.
  */
case class SG(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm(alphas)
  with hasDistributedTestSetEvaluator with hasDistributedTrainingSetEvaluator with hasGradientDescent {
	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate: SG = {
    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf, matOps), kmf.getData().getLabels(Train))
    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf, matOps), kmf.getData().getLabels(Validation))
    println(createLog(correct, misclassified, correctT, misclassifiedT, alphas))
    assert(getSparsity < 99.0)
    //Decrease the step size, i.e. learning rate:
		val ump = mp.updateDelta(ap)
		//Update the alphas using gradient descent
		gradientDescent(alphas, ap, ump, kmf)
	}

  /**
    * Performs a stochastic gradient update with clipping of the alphas.
    * Note the difference in the return type.
    * @param alphas The primal variables.
    * @param ap The properties of the algorithm.
    * @param mp The properties of the model.
    * @param kmf A MatrixFactory object.
    * @return An updated instance of the algorithm.
    */
  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): SG = {
    val stochasticUpdate = calculateGradientDescent (alphas, ap, mp, kmf)
    copy(alphas = alphas.copy(alpha = stochasticUpdate).updateAlphaAsConjugateGradient().clipAlphas(ap.quantileAlphaClipping))
  }
}