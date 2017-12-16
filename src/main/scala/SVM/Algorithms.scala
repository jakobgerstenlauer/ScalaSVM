package SVM

import breeze.linalg.{DenseVector, _}
import breeze.numerics._
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

case class AllMatrixElementsZeroException(message:String) extends Exception(message)
case class EmptyRowException(message:String) extends Exception(message)

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

abstract class Algorithm

/**
*Sequential gradient descent algorithm
**/
case class SG(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm with hasBagging with hasTestEvaluator with hasTrainingSetEvaluator{

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SG = {
		//Compute correct minus incorrect classifications on training set
		val predictions : DenseVector[Double] = evaluateOnTrainingSet(alphas, ap, kmf, matOps)
		assert(predictions.length == kmf.d.getLabelsTrain.length)
		val product : DenseVector[Double] = predictions *:* kmf.d.getLabelsTrain.map(x => x.toDouble)
		val correct = product.map(x=>if(x>0) 1 else 0).reduce(_+_)
		val misclassified : Int = product.map(x=>if(x<0) 1 else 0).reduce(_+_)
		println("Training set: "+ correct +"/"+ misclassified)

		//Compute correct minus incorrect classifications on test set
		val predictionsTest : DenseVector[Double] = evaluateOnTestSet(alphas, ap, kmf, matOps)
		assert(predictionsTest.length == kmf.d.getLabelsTest.length)
		val productT : DenseVector[Double] = predictionsTest *:* kmf.d.getLabelsTest.map(x => x.toDouble)
		val correctT = productT.map(x=>if(x>0) 1 else 0).reduce(_+_)
		val misclassifiedT : Int = productT.map(x=>if(x<0) 1 else 0).reduce(_+_)
		println("Test set: "+ correctT + "/" + misclassifiedT)

		//Decrease the step size, i.e. learning rate:
		val ump = mp.updateDelta(ap)

		//Update the alphas using gradient descent
		stochasticSequentialGradient(alphas, ap, ump, kmf)
	}

	def sequentialGradient(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory) : SG = {
    val N: Int = alphas.alpha.length
		val maxPrintIndex: Int = min(10, N)
		val gradient: DenseVector[Double] = kmf.calculateGradient(alphas.alpha)
    if(ap.isDebug){
      println("gradient: " + gradient(0 until maxPrintIndex))
    }

		//Extract model parameters
		val delta = mp.delta
		val C = mp.C

		//Extract the labels for the training set
		val d = kmf.d.getLabelsTrain.map(x => x.toDouble)
    if(ap.isDebug){
        println("alphas before update:"+alphas.alpha(0 until maxPrintIndex))
    }

		//Our first, tentative, estimate of the updated parameters is:
		val alpha1 : DenseVector[Double] = alphas.alpha - delta *:* gradient
    if(ap.isDebug){
        println("alphas first tentative update:"+alpha1(0 until maxPrintIndex))
    }

		//Then, we have to project the alphas onto the feasible region defined by the first constraint:
    val alpha2 : DenseVector[Double] = alpha1 - ((d * (d dot alpha1)) / (d dot d))

		//The value of alpha has to be between 0 and C.
    if(ap.isDebug){
        println("alphas after projection:"+alpha2(0 until maxPrintIndex))
    }
    val alpha3 : DenseVector[Double] = alpha2.map(x => if(x > C) C else x).map(x => if(x > 0) x else 0)
    if(ap.isDebug){
        println("alphas after 2nd projection:"+alpha3(0 until maxPrintIndex))
    }
    copy(alphas= alphas.copy(alpha=alpha3))
	}
	
  def stochasticSequentialGradient(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory) : SG = {
    //Extract model parameters
		val N = alphas.alpha.length
		val maxPrintIndex = min(10, N)
    val C = mp.C
		val d = kmf.d.getLabelsTrain.map(x=>x.toDouble)
    val delta = mp.delta
    val shrinking = ap.learningRateDecline
    val shrinkedValues: DenseVector[Double] = shrinking * alphas.alpha
    val gradient = kmf.calculateGradient(alphas.alpha)

    if(ap.isDebug){
        println("gradient: " + gradient(0 until maxPrintIndex))
        println("alphas before update:"+alphas.alpha(0 until maxPrintIndex))
    }

    //Our first, tentative, estimate of the updated parameters is:
		val alpha1 : DenseVector[Double] = alphas.alpha - delta *:* gradient
    if(ap.isDebug){
        println("alphas first tentative update:"+alpha1(0 until maxPrintIndex))
    }

    //Then, we have to project the alphas onto the feasible region defined by the first constraint:
    val alpha2 : DenseVector[Double] = alpha1 - (d * (d dot alpha1)) / (d dot d)
    if(ap.isDebug){
        println("alphas after projection:"+alpha2(0 until maxPrintIndex))
    }

    //The value of alpha has to be between 0 and C.
    val updated = alpha2.map(alpha => if(alpha > C) C else alpha).map(alpha => if(alpha > 0) alpha else 0.0)
    if(ap.isDebug)println("alphas after projection:"+updated(0 until maxPrintIndex))

    //random boolean vector: is a given observation part of the batch? (0:no, 1:yes)
    val batchProb : Double = ap.batchProb
    val randomUniformNumbers : DenseVector[Double] = DenseVector.rand(N)
    val isInBatch : DenseVector[Double] = randomUniformNumbers.map(x => if(x < batchProb) 1.0 else 0.0)
    if(ap.isDebug) println("batch vector: "+isInBatch(0 until maxPrintIndex))

    val tuples = (isInBatch.toArray.toList zip shrinkedValues.toArray.toList zip updated.toArray.toList) map { case ((a,b),c) => (a,b,c)}
    val stochasticUpdate = new DenseVector(tuples.map{ case (inBatch, alphas_shrinked, alphas_updated) => if (inBatch == 1) alphas_updated else alphas_shrinked}.toArray)
    if(ap.isDebug)println("stochastic update:"+stochasticUpdate(0 until maxPrintIndex))
    copy(alphas= alphas.copy(alpha=stochasticUpdate))
	}
}

/**
*Sequential gradient descent algorithm
**/
case class SGtest(alphas: Alphas, ap: AlgoParams, mp: ModelParams, lkmf: LocalKernelMatrixFactory) extends Algorithm with hasBagging with hasTestEvaluator {

	def iterate() : SGtest = {
                //Decrease the step size, i.e. learning rate:
		val ump = mp.updateDelta(ap)

		//Update the alphas using gradient descent
  		val algo = sequentialGradient(alphas, ap, ump, lkmf)
		
                algo
        }

	def sequentialGradient(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: LocalKernelMatrixFactory) : SGtest = {
		val gradient = kmf.calculateGradient(alphas.alpha)
		//Extract model parameters
		val delta = mp.delta
		val C = mp.C
		//Extract the labels for the training set
		val d = kmf.d.getLabelsTrain.map(x=>x.toDouble)
    //Our first, tentative, estimate of the updated parameters is:
		val alpha1 : DenseVector[Double] = alphas.alpha - delta *:* gradient
    //Then, we have to project the alphas onto the feasible region defined by the first constraint:
    val alpha2 : DenseVector[Double] = alpha1 - (d * (d dot alpha1)) / (d dot d)
    //The value of alpha has to be between 0 and C.
    val alpha3 : DenseVector[Double] = alpha2.map(x => if(x > C) C else x).map(x => if(x > 0) x else 0)
    copy(alphas= alphas.copy(alpha=alpha3))
	}
}

/**
*Stochastic gradient descent algorithm
**/
case class SGD(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm with hasBagging with hasTestEvaluator with hasTrainingSetEvaluator{

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SGD = {

		//Compute correct minus incorrect classifications on training set
    val predictionsTrain = evaluateOnTrainingSet(alphas, ap, kmf, matOps)
    val productTrain : DenseVector[Double] = predictionsTrain *:* kmf.d.getLabelsTrain.map(x => x.toDouble)
    val correctTrain : Int = productTrain.map(x=>if(x>0) 1 else 0).reduce(_+_)
    val misclassifiedTrain : Int = productTrain.map(x=>if(x<0) 1 else 0).reduce(_+_)
    println("Training set: "+ correctTrain +"/"+ misclassifiedTrain)
		
    //Compute correct minus incorrect classifications on test set
    val predictions = evaluateOnTestSet(alphas, ap, kmf, matOps)
    val product : DenseVector[Double] = predictions *:* kmf.d.getLabelsTest.map(x => x.toDouble)
    val correct : Int = product.map(x => if(x>0) 1 else 0).reduce(_+_)
    val misclassified : Int = product.map(x => if(x<0) 1 else 0).reduce(_+_)
    println("Test set: "+ correct +"/"+ misclassified)
    println("Delta alpha: " + alphas.getDelta )

    //Decrease the step size, i.e. learning rate:
    val ump = mp.updateDelta(ap)

    //Create a random sample of alphas and store it in a distributed matrix Alpha:
    val Alpha = getDistributedAlphas(ap, alphas, kmf, sc)

    //Update the alphas using gradient descent
    val algo = gradientDescent(Alpha, alphas, ap, ump, kmf, matOps)

    algo
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
		//val max_print_index = max(10, kmf.d.getd)

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
		val z = kmf.d.getLabelsTrain.map(x=>x.toDouble)
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

class Norma(alphas: Alphas, ap:AlgoParams, mp:ModelParams, kmf:KernelMatrixFactory, sc: SparkContext) 
	extends SGD(alphas, ap, mp, kmf, sc) 

class Silk(alphas: Alphas, ap:AlgoParams, mp:ModelParams, kmf:KernelMatrixFactory, sc: SparkContext)
	extends SGD(alphas, ap, mp, kmf, sc) 


trait hasTrainingSetEvaluator extends Algorithm{
	/**
	* Returns the number of correct predictions minus the nr of misclassifications for a test set.
	*
	* alphas: The alpha parameters.
	* ap:     AlgoParams object storing parameters of the algorithm
	* kmf:    KernelMatrixFactory that contains the distributed matrices for the data set
	* matOps: A matrix operations object 
	***/
	def evaluateOnTrainingSet(alphas: Alphas, ap: AlgoParams, kmf: KernelMatrixFactory, matOps: DistributedMatrixOps):DenseVector[Double]= {

		//Get the distributed kernel matrix for the test set:
		val K = kmf.K
		val z = kmf.z.map(x=>x.toDouble)
		val epsilon = max(min(ap.epsilon, min(alphas.alpha)), 0.000001)
		val A = matOps.distributeRowVector(alphas.alpha *:* z, epsilon)

 		assert(z!=null && A!=null && K!=null, "One of the input matrices is undefined!")
    assert(A.numCols()>0, "The number of columns of A is zero.")
    assert(A.numRows()>0, "The number of rows of A is zero.")
    assert(K.numCols()>0, "The number of columns of S is zero.")
    assert(K.numRows()>0, "The number of rows of S is zero.")
    assert(A.numCols()==K.numRows(),"The number of columns of A does not equal the number of rows of S!")
    assert(K.numCols()==z.length,"The number of columns of S does not equal the number of rows of Z!")

    if(ap.isDebug){
      println("K:")
      K.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
      println()
      println("Z:")
      println(z)
      println()
      println("alphas:")
      println(alphas.alpha)
      println()
      println("A:")
      A.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
    }

		val P = matOps.coordinateMatrixMultiply(A, K)
    if(ap.isDebug){
      println("predictions:")
      P.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
    }

    //Return the predictions
    signum(matOps.collectRowVector(P))
	}
}

trait hasTestEvaluator extends Algorithm{
	/**
	* Returns the number of correct predictions minus the nr of misclassifications for a test set.
	*
	* alphas: The alpha parameters.
	* ap:     AlgoParams object storing parameters of the algorithm
	* kmf:    KernelMatrixFactory that contains the distributed matrices for the data set
	* matOps: A matrix operations object 
	***/
	def evaluateOnTestSet(alphas: Alphas, ap: AlgoParams, kmf: KernelMatrixFactory, matOps: DistributedMatrixOps) : DenseVector[Double] = {

		//Get the distributed kernel matrix for the test set:
		val S = kmf.S
		val z = kmf.z_test
		val epsilon = max(min(ap.epsilon, min(alphas.alpha)), 0.000001)
		val A = matOps.distributeRowVector(alphas.alpha *:* kmf.d.getLabelsTrain.map(x=>x.toDouble), epsilon)

 		assert(z!=null && A!=null && S!=null, "One of the input matrices is undefined!")
    assert(A.numCols()>0, "The number of columns of A is zero.")
    assert(A.numRows()>0, "The number of rows of A is zero.")
    assert(S.numCols()>0, "The number of columns of S is zero.")
    assert(S.numRows()>0, "The number of rows of S is zero.")
    assert(A.numCols()==S.numRows(),"The number of columns of A does not equal the number of rows of S!")
    assert(S.numCols()==z.length,"The number of columns of S does not equal the number of rows of Z!")

    if(ap.isDebug){
      println("S:")
      println("rows:"+S.numRows()+" columns: "+S.numCols())
      S.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
      println()
      println("z:")
      println(z)
      println()
      println("alphas:")
      println(alphas.alpha)
      println()
      println("A:")
      A.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
    }

    val P = matOps.coordinateMatrixMultiply(A, S)
    if(ap.isDebug){
            println("predictions:")
            println("rows:"+P.numRows()+" columns: "+P.numCols())
            P.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
            println("Length collected vector of predictions:" + matOps.collectRowVector(P).length)
            println("Content:"+matOps.collectRowVector(P))

    }
    //Return the predictions
    signum(matOps.collectRowVector(P))
	}
}

trait hasBagging extends Algorithm{

	def getDistributedAlphas(ap: AlgoParams, alphas: Alphas, kmf: KernelMatrixFactory, sc: SparkContext) : CoordinateMatrix = {
		val batchMatrix = getBatchMatrix(ap, kmf)
		createStochasticGradientMatrix(alphas, batchMatrix, ap.epsilon, sc)
	}

	private def getBatchMatrix(ap: AlgoParams, kmf: KernelMatrixFactory) : DenseMatrix[Double] = {
    val randomMatrix : DenseMatrix[Double] = DenseMatrix.rand(ap.numBaggingReplicates, kmf.d.getN_train)
		val batchMatrix = randomMatrix.map(x=>if(x < ap.batchProb) 1.0 else 0.0)
    batchMatrix
	}

 	private	def exceeds(x: Double, e: Double) : Boolean = abs(x) > e

	private def createStochasticGradientMatrix(alphas: Alphas, m: DenseMatrix[Double], epsilon: Double, sc: SparkContext) : CoordinateMatrix = {
		val a = alphas.alpha
		val a_old = alphas.alphaOld
		assert(epsilon > 0, "The value of epsilon must be positive!")
    assert(a.length > 0, "The input vector with the alphas is empty!!!")
    assert(m.rows > 0, "The dense matrix m must have at least 1 row!!!")
    assert(m.cols == a.length, "The number of columns of the matrix m("+m.cols+") must be equal to the length of alpha("+a.length+")!!!")
		//If entry i,j of the matrix m is 1 we set element i,j of A to a else a_old:
    val listOfMatrixEntries =  for (i <- 0 until m.rows; j <- 0 until a.length) yield MatrixEntry(i, j, m(i,j)*a(j)+(1-m(i,j))*a_old(j))
    // Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries.filter(x => exceeds(x.value,epsilon)))
    //entries.collect().map({ case MatrixEntry(row, column, value) => println("row: "+row+" column: "+column+" value: "+value)})
    if(entries.count()==0) throw allAlphasZeroException("All values of the distributed matrix are zero!")
    // Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
    new CoordinateMatrix(entries, m.rows, a.length.toLong)
	}
}
