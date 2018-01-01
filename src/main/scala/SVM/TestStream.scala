package SVM

object TestStream extends App{
  case class Indices(i: Int, j: Int)
  def streamMatrixIndices(m: Indices, numCols: Int): Stream[Indices] = {
    if(m.j<(numCols-1)) m #:: streamMatrixIndices(Indices(m.i,m.j+1),numCols) else{
      m #:: streamMatrixIndices(Indices(m.i+1,m.i+2),numCols)
    }
  }
  def getMatrixIndexStream(numCols: Int): Stream[Indices] =
  {
    val NumElements: Int = ((numCols * numCols) - numCols) / 2
    streamMatrixIndices(Indices(0, 1), numCols).take(NumElements)
  }
  getMatrixIndexStream(5).toList.foreach(println)
}
