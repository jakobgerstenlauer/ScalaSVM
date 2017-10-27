package SVM

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps
import breeze.linalg.operators
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

case class Alphas(var alpha: DenseVector[Double], var alpha_old: DenseVector[Double]){
	assert(alpha.length == alpha_old.length)
	def getDelta():Double = sum(abs(alpha - alpha_old))
}

abstract class Algorithm
case class Norma(var alphas: Alphas, val ap: AlgoParams, val mp: ModelParams, val kmf: KernelMatrixFactory) extends Algorithm 
	with decreasingLearningRate with hasMomentum with hasBagging
 
case class Silk(var alphas: Alphas, val ap: AlgoParams, val mp: ModelParams, val kmf: KernelMatrixFactory) extends Algorithm
	with decreasingLearningRate with hasMomentum with hasBagging

trait decreasingLearningRate extends Algorithm{
	def decreaseLearningRate(ap: AlgoParams, mp: ModelParams):Unit={
		mp.updateDelta(ap.learningRateDecline)
	}
}

trait hasMomentum extends Algorithm{
    	//Updates the momentum according to the method of Polak-Ribiere
	def getConjugateGradient(a: Alphas) : Double = {
    		assert(a.alpha.length == a.alpha_old.length)
		val diff = a.alpha - a.alpha_old
    		assert(a.alpha.length == diff.length)
    		val dot_product = a.alpha.t * diff
    		var momentum = 0.0
		val alpha_old_norm = sqrt(a.alpha_old.map(x => pow(x,2)).reduce(_ + _))
		if(alpha_old_norm > 0.000001){ 
			momentum = dot_product / alpha_old_norm
		}
    		return a.alpha + momentum * a.alpha_old
	}
}

trait hasBagging extends Algorithm{

	def getDistributedAlphas(ap: AlgoParams, alphas: Alphas, kmf: KernelMatrixFactory, sc: SparkContext) : CoordinateMatrix = {
		val batchMatrix = getBatchMatrix(ap, kmf)
		return createStochasticGradientMatrix(alphas.alpha, batchMatrix, ap.epsilon, sc)
	}

	def getBatchMatrix(ap: AlgoParams, kmf: KernelMatrixFactory) : DenseMatrix[Double] = {
		return DenseMatrix.rand(ap.numBaggingReplicates, kmf.getData().getN_train()).map(x=>if(x < ap.batchProb) 1.0 else 0.0)
	}

	def createStochasticGradientMatrix(a: DenseVector[Double], m: DenseMatrix[Double], epsilon: Double, sc: SparkContext) : CoordinateMatrix = {
    		assert(epsilon > 0, "The value of epsilon must be positive!")
    		assert(a.length > 0, "The input vector with the alphas is empty!!!")
    		assert(m.rows > 0, "The dense matrix m must have at least 1 row!!!")
    		assert(m.cols == a.length, "The number of columns of the matrix m must be equal to the length of alpha!!!")
  
    		def exceeds(x: Double, e: Double) : Boolean = {
      			val abs_x = abs(x)
      			return abs_x > e      
    		}

    		val listOfMatrixEntries =  for (i <- 0 until m.rows; j <- 0 until a.length) yield (new MatrixEntry(i, j, m(i,j) * a(j)))
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
