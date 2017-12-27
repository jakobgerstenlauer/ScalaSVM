package SVM

import breeze.linalg.{DenseVector, max, min}
import breeze.numerics.signum
import org.apache.spark.mllib.linalg.distributed.MatrixEntry

trait hasDistributedTestSetEvaluator extends Algorithm{
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
    val epsilon = max(min(ap.epsilon, min(alphas.alpha)), 0.000001)
    val A = matOps.distributeRowVector(alphas.alpha *:* kmf.getData().getLabelsTrain.map(x=>x.toDouble), epsilon)

    assert(A!=null && S!=null, "One of the input matrices is undefined!")
    assert(A.numCols()>0, "The number of columns of A is zero.")
    assert(A.numRows()>0, "The number of rows of A is zero.")
    assert(S.numCols()>0, "The number of columns of S is zero.")
    assert(S.numRows()>0, "The number of rows of S is zero.")
    assert(A.numCols()==S.numRows(),"The number of columns of A does not equal the number of rows of S!")

    if(ap.isDebug){
      println("S:")
      println("rows:"+S.numRows()+" columns: "+S.numCols())
      S.entries.collect().foreach({ case MatrixEntry(row, column, value) => println("i: "+row+"j: "+column+": "+value)})
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

