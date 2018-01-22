package SVM

import breeze.linalg.{DenseVector, max, min}
import breeze.numerics.signum
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

trait hasDistributedTrainingSetEvaluator extends Algorithm{
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
    val K : CoordinateMatrix = kmf.K
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