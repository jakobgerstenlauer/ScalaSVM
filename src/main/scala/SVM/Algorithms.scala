package SVM

import breeze.linalg.{DenseVector, _}
import org.apache.spark.SparkContext
case class AllMatrixElementsZeroException(message:String) extends Exception(message)
case class EmptyRowException(message:String) extends Exception(message)

abstract class Algorithm{
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
}

/**
  * Lean implementation of the sequential gradient descent algorithm without matrices.
  * @param alphas The current and old values of the alphas.
  * @param ap Properties of the algorithm
  * @param mp Properties of the model
  * @param kmf A KernelMatrixFactory for local matrices.N
  */
case class NoMatrices(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LeanMatrixFactory) extends Algorithm with hasGradientDescent {

  def iterate() : NoMatrices = {

    val (correct, misclassified) = calculateAccuracy(kmf.predictOnTrainingSet(alphas.alpha), kmf.getData().getLabelsTrain)
    val (correctT, misclassifiedT) = calculateAccuracy(kmf.predictOnTestSet(alphas.alpha), kmf.getData().getLabelsTest)
    val sparsity = 1.0 - alphas.alpha.map(x=>if (x>0) 1 else 0).reduce(_+_).toDouble / alphas.alpha.length.toDouble
    println("Train: "+ correct +"/"+ misclassified + ", Test: "+ correctT + "/" + misclassifiedT+ ", Sparsity: "+ sparsity)

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
  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LeanMatrixFactory): NoMatrices = {
    val stochasticUpdate = calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory)
    copy(alphas = alphas.copy(alpha = stochasticUpdate))
  }
}

/**
  * Sequential gradient descent algorithm with local matrices
  * @param alphas The current and old values of the alphas.
  * @param ap Properties of the algorithm
  * @param mp Properties of the model
  * @param kmf A KernelMatrixFactory for local matrices.
  */
case class SGLocal(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LocalKernelMatrixFactory) extends Algorithm
  with hasLocalTrainingSetEvaluator with hasLocalTestSetEvaluator with hasGradientDescent {

  def iterate() : SGLocal = {

    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf), kmf.getData().getLabelsTrain)
    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf), kmf.getData().getLabelsTest)
    val sparsity = 1.0 - alphas.alpha.map(x=>if (x>0) 1 else 0).reduce(_+_).toDouble / alphas.alpha.length.toDouble
    println("Train: "+ correct +"/"+ misclassified + ", Test: "+ correctT + "/" + misclassifiedT+ ", Sparsity: "+ sparsity)

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
    val stochasticUpdate = calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory)
    copy(alphas = alphas.copy(alpha = stochasticUpdate))
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
case class SG(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm
  with hasDistributedTestSetEvaluator with hasDistributedTrainingSetEvaluator with hasGradientDescent {

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SG = {

    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTrain)
    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTest)
    val sparsity = 1.0 - alphas.alpha.map(x=>if (x>0) 1 else 0).reduce(_+_).toDouble / alphas.alpha.length.toDouble
    println("Train: "+ correct +"/"+ misclassified + ", Test: "+ correctT + "/" + misclassifiedT+ ", Sparsity: "+ sparsity)

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
    val stochasticUpdate = calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory)
    copy(alphas = alphas.copy(alpha = stochasticUpdate))
  }
}