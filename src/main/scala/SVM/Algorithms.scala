package SVM

import breeze.linalg.{DenseVector, _}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.SparkContext
case class AllMatrixElementsZeroException(message:String) extends Exception(message)
case class EmptyRowException(message:String) extends Exception(message)

abstract class Algorithm{
  def iterate : Algorithm

  def calculateAccuracy(predictions: DenseVector[Double], labels: DenseVector[Int]):(Int,Int) = {
    assert(predictions.length == labels.length)
    val product : DenseVector[Double] = predictions *:* labels.map(x => x.toDouble)
    val correct = product.map(x=>if(x>0) 1 else 0).reduce(_+_)
    val misclassified : Int = product.map(x=>if(x<0) 1 else 0).reduce(_+_)
    (correct, misclassified)
  }
}

/**
  *Sequential gradient descent algorithm with local matrices
  **/
case class SGLocal(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LocalKernelMatrixFactory) extends Algorithm
  with hasLocalTrainingSetEvaluator with hasLocalTestSetEvaluator with hasGradientDescent {

  def iterate() : SGLocal = {

    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf), kmf.getData().getLabelsTrain)
    println("Training set: "+ correct +"/"+ misclassified)

    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf), kmf.getData().getLabelsTest)
    println("Test set: "+ correctT + "/" + misclassifiedT)

    //Decrease the step size, i.e. learning rate:
    val ump = mp.updateDelta(ap)

    //Update the alphas using gradient descent
    gradientDescent(alphas, ap, ump, kmf)
  }

  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): SGLocal = {
    val stochasticUpdate = calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory)
    copy(alphas = alphas.copy(alpha = stochasticUpdate))
  }
}

/**
*Sequential gradient descent algorithm with distributed matrices
**/
case class SG(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm
  with hasDistributedTestSetEvaluator with hasDistributedTrainingSetEvaluator with hasGradientDescent {

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SG = {

    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTrain)
    println("Training set: "+ correct +"/"+ misclassified)

    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTest)
    println("Test set: "+ correctT + "/" + misclassifiedT)

		//Decrease the step size, i.e. learning rate:
		val ump = mp.updateDelta(ap)

		//Update the alphas using gradient descent
		gradientDescent(alphas, ap, ump, kmf)
	}

  def gradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory): SG = {
    val stochasticUpdate = calculateGradientDescent (alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: MatrixFactory)
    copy(alphas = alphas.copy(alpha = stochasticUpdate))
  }
}

/**
*Stochastic gradient descent algorithm
**/
case class SGD(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm with hasBagging with hasDistributedTestSetEvaluator with hasDistributedTrainingSetEvaluator{

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SGD = {

    val (correct, misclassified) = calculateAccuracy(evaluateOnTrainingSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTrain)
    println("Training set: "+ correct +"/"+ misclassified)

    val (correctT, misclassifiedT) = calculateAccuracy(evaluateOnTestSet(alphas, ap, kmf, matOps), kmf.getData().getLabelsTest)
    println("Test set: "+ correctT + "/" + misclassifiedT)

    //Decrease the step size, i.e. learning rate:
    val ump = mp.updateDelta(ap)

    //Create a random sample of alphas and store it in a distributed matrix Alpha:
    val Alpha = getDistributedAlphas(ap, alphas, kmf, sc)

    //Update the alphas using gradient descent
    gradientDescent(Alpha, alphas, ap, ump, kmf, matOps)
  }

	def gradientDescent(Alpha: CoordinateMatrix, alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, matOps: DistributedMatrixOps) : SGD = {

		//Get the distributed kernel matrix for the training set:
		val K = kmf.K
    assert(Alpha.numCols()==K.numRows(),"The number of columns of Alpha does not equal the number of rows of K!")

		//Calculate the predictions for all replicates (rows of Alpha) with different batches of observations.
		//Note that elements alpha_i are zero for all observations that are not part of a given batch.
		val predictionsMatrix = matOps.coordinateMatrixMultiply(Alpha, K)

		//Make sure, that there is at least one non-zero matrix element in the matrix:
		val nonSparseElements = predictionsMatrix.entries.count()
    if(nonSparseElements==0){
        throw allAlphasZeroException("All values of alpha are zero!")
    }

    if(ap.isDebug){
            println("predictions:")
            predictionsMatrix.entries.collect().map({ case MatrixEntry(row, _, value) => (row,value)}).groupBy(_._1).mapValues(_.unzip._2.sum).foreach(println)
    }

    //Calculate the vector of length batch replicates whose elements represent the nr of misclassifications:
    //TODO Implement same logic as in evaluateOnTrainingSet() !!!
    val Z = kmf.Z
		val errorMatrix = matOps.coordinateMatrixSignumAndMultiply(predictionsMatrix, Z)

    //Find the index with the smallest error and use these alphas:
    assert(errorMatrix.entries.count()>0,"No elements in errorMatrix!")

    //key: nr of correct classifications - nr of misclassifications
    //value: row idex
		val precision_index_map = errorMatrix.entries.map({ case MatrixEntry(row, column, value) => (value,row) })

    assert(precision_index_map.count()>0,"No elements in precision_index_map!")

    if(ap.isDebug) errorMatrix.entries.map({ case MatrixEntry(row, _, value) => println("correct classifications: "+value+" row "+row) })

    //Sort the map according to the accuracy of the given coefficients alpha and get the first precision index pair:
    //Compare: http://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-operations
    val sortedPrecisionMap = precision_index_map.sortByKey(ascending=false)

    assert(sortedPrecisionMap.count()>0,"No elements in sortedPrecisionMap!")

    val (_, index) = sortedPrecisionMap.first()

		//Retrieve the vector of alphas for this row index
		val alphas_opt_optional = matOps.getRow(Alpha, index.toInt)
		val alphas_opt = alphas_opt_optional.getOrElse(throw EmptyRowException("Row "+index.toInt+" of matrix Alpha is empty!"))
    val isInBatch: DenseVector[Int] = alphas_opt.map(x => if(x>0) 1 else 0)

 		//Retrieve the vector of predictions for this row index
		val prediction_optional = matOps.getRow(predictionsMatrix, index.toInt)
		val prediction = prediction_optional.getOrElse(throw EmptyRowException("Row "+index.toInt+" of the prediction matrix is empty!"))

		//Extract model parameters
		val lambda = mp.lambda
		val delta = mp.delta
		val C = mp.C
		//Extract the labels for the training set
		val z = kmf.getData().getLabelsTrain.map(x=>x.toDouble)
		val shrinking = 1 - lambda * delta
		val tau = (lambda * delta)/(1 + lambda * delta)
		val shrinkedValues = shrinking * alphas.alphaOld
		val deviance =  prediction *:* z

		//Compare the different update formulas for soft margin and hinge-loss in "Online Learning with Kernels", Kivinen, Smola, Williamson (2004)
		//http://realm.sics.se/papers/KivSmoWil04(1).pdf
		val term1 = (1.0 - tau) * deviance
		val term2 : DenseVector[Double]= DenseVector.ones[Double](z.length) - term1
		val alpha_hat : DenseVector[Double] = z *:* term2
		val tuples_alpha_y = alpha_hat.toArray.toList zip z.toArray.toList
		val threshold : Double = (1.0 - tau) * C
		val updated = tuples_alpha_y.map{ case (alpha_hat_, y) => if (y * alpha_hat_ < 0.0) 0.0 else {
      if (y * alpha_hat_ > threshold) y * threshold else alpha_hat_
    }}.toArray
		val tuples = (isInBatch.toArray.toList zip shrinkedValues.toArray.toList zip updated.toList) map { case ((a,b),c) => (a,b,c)}
		val new_alphas = new DenseVector(tuples.map{ case (inBatch, alphas_shrinked, alphas_updated) => if (inBatch == 1) alphas_updated else alphas_shrinked}.toArray)
    val new_Alpha = if(ap.hasMomentum) alphas.copy(alpha=new_alphas).updateAlphaAsConjugateGradient() else alphas.copy(alpha=new_alphas)
    copy(alphas= new_Alpha)
	}
}




