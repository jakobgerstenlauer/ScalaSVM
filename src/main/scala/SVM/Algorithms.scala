package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps
import breeze.linalg.operators
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

case class AllMatrixElementsZeroException(smth:String) extends Exception(smth)
case class EmptyRowException(smth:String) extends Exception(smth)

/**
* N: The number of observations in the training set
**/
case class Alphas(N: Int, 
  alpha: DenseVector[Double] = DenseVector.ones[Double](N) - DenseVector.rand(N),
  alphaOld: DenseVector[Double] = DenseVector.zeros[Double](N)
  ){
	def getDelta() : Double = sum(abs(alpha - alphaOld))
	
        def updateAlphaAsConjugateGradient() : Alphas = {
                val diff = alpha - alphaOld
                val dotProduct = alpha.t * diff
                val alphaOldNorm = sqrt(alphaOld.map(x => pow(x,2)).reduce(_ + _))
                if(alphaOldNorm > 0.000001){
                    val momentum = dotProduct / alphaOldNorm
                    val alphaUpdated = alpha + momentum * alphaOld
                    val alphaUpdatedNorm = sqrt(alphaUpdated.map(x => pow(x,2)).reduce(_ + _))
                    //Return a copy of this object with alpha updated according to the
                    //Polak-Ribiere conjugate gradient formula.
                    //Compare: https://en.wikipedia.org/wiki/Conjugate_gradient_method
		    copy(alpha = alphaUpdated / alphaUpdatedNorm, alphaOld = alpha)
                }else{
                    //If the norm of alpha in the previous step is below a threshold,
                    //return a copy of this object without any changes.
                    copy()
                }
        }
}

abstract class Algorithm

/**
*Stochastic gradient descent algorithm
**/
case class SGD(alphas: Alphas, ap: AlgoParams, mp: ModelParams, kmf: KernelMatrixFactory, sc: SparkContext) extends Algorithm with hasMomentum with hasBagging with hasGradientDescent with hasTestEvaluator{

	val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

	def iterate() : SGD = {

		//Decrease the step size, i.e. learning rate:
		val ump = mp.updateDelta(ap.learningRateDecline)

		//Calculate the conjugate gradient
		val updatedAlphas = updateConjugateGradient(alphas)			

		//Create a random sample of alphas and store it in a distributed matrix Alpha:
		val Alpha = getDistributedAlphas(ap, updatedAlphas, kmf, sc)

		//Update the alphas using gradient descent
  		val algo = gradientDescent(Alpha, updatedAlphas, ap, ump, kmf, matOps)
		
		//Compute correct minus incorrect classifications on test set
		val predictionQuality = evaluateOnTestSet(updatedAlphas, ap, kmf, matOps)
		
		val delta_alpha = updatedAlphas.getDelta()
  		println("Prediction quality test: "+ predictionQuality + " delta alpha: " + delta_alpha)
	        
                algo
        }
}

class Norma(alphas: Alphas, ap:AlgoParams, mp:ModelParams, kmf:KernelMatrixFactory, sc: SparkContext) 
	extends SGD(alphas, ap, mp, kmf, sc) 

class Silk(alphas: Alphas, ap:AlgoParams, mp:ModelParams, kmf:KernelMatrixFactory, sc: SparkContext)
	extends SGD(alphas, ap, mp, kmf, sc) 

trait hasTestEvaluator extends Algorithm{
	/**
	* Returns the number of correct predictions minus the nr of misclassifications for a test set.
	*
	* alphas: The alpha parameters.
	* ap:     AlgoParams object storing parameters of the algorithm
	* kmf:    KernelMatrixFactory that contains the distributed matrices for the data set
	* matOps: A matrix operations object 
	***/
	def evaluateOnTestSet(alphas: Alphas, ap: AlgoParams, kmf: KernelMatrixFactory, matOps: DistributedMatrixOps) : Int = {

		//Get the distributed kernel matrix for the test set:
		val S = kmf.S
		val Z = kmf.Z_test
		val epsilon = max(ap.epsilon, min(alphas.alpha))
		val A = matOps.distributeRowVector(alphas.alpha, epsilon)

 		assert(Z!=null && A!=null && S!=null, "One of the input matrices is undefined!")
  		assert(A.numCols()>0, "The number of columns of A is zero.")
  		assert(A.numRows()>0, "The number of rows of A is zero.")
  		assert(S.numCols()>0, "The number of columns of S is zero.")
  		assert(S.numRows()>0, "The number of rows of S is zero.")
  		assert(A.numCols()==S.numRows(),"The number of columns of A does not equal the number of rows of S!")
  		assert(S.numCols()==Z.numRows(),"The number of columns of S does not equal the number of rows of Z!")  

		val P = matOps.coordinateMatrixMultiply(A, S)
		val E = matOps.coordinateMatrixSignumAndMultiply(P, Z)

		//This a matrix with only one entry which we retrieve with first():
		return E.entries.map({ case MatrixEntry(i, j, v) => v }).first().toInt
	}
}

trait hasGradientDescent extends Algorithm{

	def printVector(vector: DenseVector[Double], max_index: Int, label: String) : Unit = {
		println(label + " (1 to "+max_index+" ):"+ vector(0 until max_index))
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
    			throw new allAlphasZeroException("All values of alpha are zero!")
  		}  

                //predictionsMatrix.entries.collect().map({ case MatrixEntry(row, column, value) => (row,value)}).groupBy(_._1).mapValues(_.unzip._2.sum).map(println)
		
                //Calculate the vector of length batch replicates whose elements represent the nr of misclassifications: 
  		val Z = kmf.Z
		val errorMatrix = matOps.coordinateMatrixSignumAndMultiply(predictionsMatrix, Z)

  		//Find the index with the smallest error and use these alphas:
  		assert(errorMatrix.entries.count()>0,"No elements in errorMatrix!")
  		//key: nr of correct classifications - nr of misclassifications
  		//value: row idex  
		val precision_index_map = errorMatrix.entries.map({ case MatrixEntry(row, column, value) => (value,row) })
		assert(precision_index_map.count()>0,"No elements in precision_index_map!")
		
		//Sort the map according to the accuracy of the given coefficients alpha and get the first precision index pair:
  		//Compare: http://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-operations
		val sortedPrecisionMap = precision_index_map.sortByKey(ascending=false)
		assert(sortedPrecisionMap.count()>0,"No elements in sortedPrecisionMap!")
		val (precision, index) = sortedPrecisionMap.first()
		val max_index = max(10, kmf.getData().getd())

		//Retrieve the vector of alphas for this row index
		val alphas_opt_optional = matOps.getRow(Alpha, index.toInt)
		val alphas_opt = alphas_opt_optional.getOrElse(throw new EmptyRowException("Row "+index.toInt+" of matrix Alpha is empty!")) 
  		val isInBatch = alphas_opt.map(x => if(x>0) 1 else 0) 

 		//Retrieve the vector of predictions for this row index
		val prediction_optional = matOps.getRow(predictionsMatrix, index.toInt)
		val prediction = prediction_optional.getOrElse(throw new EmptyRowException("Row "+index.toInt+" of the prediction matrix is empty!")) 
 		
		//Extract model parameters
		val lambda = mp.lambda 
		val delta = mp.delta
		val C = mp.C
		//Extract the labels for the training set
		val z = kmf.getData().z_train
		
		val shrinking = 1 - lambda * delta  
		val tau = (lambda * delta)/(1 + lambda * delta)
		val shrinkedValues = shrinking * alphas.alphaOld
		val deviance =  prediction :* z
  
		//Compare the different update formulas for soft margin and hinge-loss in "Online Learning with Kernels", Kivinen, Smola, Williamson (2004)
		//http://realm.sics.se/papers/KivSmoWil04(1).pdf
		val term1 = (1.0 - tau) * deviance
		val term2 = DenseVector.ones[Double](z.length) - term1
		val alpha_hat = z :* term2 
		val tuples_alpha_y = (alpha_hat.toArray.toList zip z.toArray.toList)
		val threshold = (1.0 - tau) * C
		val updated = tuples_alpha_y.map{ case (alpha_hat, y) => if (y * alpha_hat < 0.0 ) 0.0 else 
                                                   if(y * alpha_hat > threshold) y * threshold else alpha_hat}.toArray
		val tuples = (isInBatch.toArray.toList zip shrinkedValues.toArray.toList zip updated.toList) map { case ((a,b),c) => (a,b,c)}
		val new_alphas = new DenseVector(tuples.map{ case (isInBatch, shrinkedValues, updated) => if (isInBatch == 1) updated else shrinkedValues}.toArray)
                val new_Alpha = Alphas.copy(alpha=new_alphas).updateAlphaAsConjugateGradient()
                SGD()
	}
}

trait hasMomentum extends Algorithm{
    	//Updates the momentum according to the method of Polak-Ribiere
	def updateConjugateGradient(alphas: Alphas) : Alphas = {
		alphas.updateAlphaAsConjugateGradient()
	}
}

trait hasBagging extends Algorithm{

	def getDistributedAlphas(ap: AlgoParams, alphas: Alphas, kmf: KernelMatrixFactory, sc: SparkContext) : CoordinateMatrix = {
		val batchMatrix = getBatchMatrix(ap, kmf)
		return createStochasticGradientMatrix(alphas, batchMatrix, ap.epsilon, sc)
	}

	private def getBatchMatrix(ap: AlgoParams, kmf: KernelMatrixFactory) : DenseMatrix[Double] = {
		return DenseMatrix.rand(ap.numBaggingReplicates, kmf.getData().getN_train()).map(x=>if(x < ap.batchProb) 1.0 else 0.0)
	}

  
    	private	def exceeds(x: Double, e: Double) : Boolean = {
      		val abs_x = abs(x)
      		return abs_x > e      
    	}

	private def createStochasticGradientMatrix(alphas: Alphas, m: DenseMatrix[Double], epsilon: Double, sc: SparkContext) : CoordinateMatrix = {
    		
		val a = alphas.alpha
		val a_old = alphas.alpha_old
		assert(epsilon > 0, "The value of epsilon must be positive!")
    		assert(a.length > 0, "The input vector with the alphas is empty!!!")
    		assert(m.rows > 0, "The dense matrix m must have at least 1 row!!!")
    		assert(m.cols == a.length, "The number of columns of the matrix m("+m.cols+") must be equal to the length of alpha("+a.length+")!!!")

		//If entry i,j of the matrix m is 1 we set element i,j of A to a else a_old:
    		val listOfMatrixEntries =  for (i <- 0 until m.rows; j <- 0 until a.length) yield (new MatrixEntry(i, j, m(i,j)*a(j)+(1-m(i,j))*a_old(j)))
    		// Create an RDD of matrix entries ignoring all matrix entries which are smaller than epsilon.
    		val entries: RDD[MatrixEntry] = sc.parallelize(listOfMatrixEntries.filter(x => exceeds(x.value,epsilon)))
    		//entries.collect().map({ case MatrixEntry(row, column, value) => println("row: "+row+" column: "+column+" value: "+value)})
 
    		if(entries.count()==0){
      			throw new allAlphasZeroException("All values of the distributed matrix are zero!")
    		}  
  
    		// Create a distributed CoordinateMatrix from an RDD[MatrixEntry].
    		return new CoordinateMatrix(entries, m.rows, a.length.toLong)
	}
}
