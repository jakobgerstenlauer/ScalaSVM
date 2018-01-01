package SVM

case class Indices(i: Int, j: Int)

object MatrixIndexStream {

  def streamMatrixIndices(m: Indices, numCols: Int): Stream[Indices] = {
    if(m.j<(numCols-1)) m #:: streamMatrixIndices(Indices(m.i,m.j+1),numCols) else{
      m #:: streamMatrixIndices(Indices(m.i+1,m.i+2),numCols)
    }
  }

  /**
    * Returns a stream of the indexes of the upper diagonal square matrix with numCols columns.
    * @param numCols Number of columns of the matrix.
    * @return
    */
  def getMatrixIndexStream(numCols: Int): Stream[Indices] =
  {
    val NumElements: Int = ((numCols * numCols) - numCols) / 2
    streamMatrixIndices(Indices(0, 1), numCols).take(NumElements)
  }
}
