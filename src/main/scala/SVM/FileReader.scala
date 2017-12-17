package SVM
import breeze.linalg._

import scala.io.{BufferedSource, Source}
abstract class FileReader(path: String)

class CSVReader(path: String, separator: Char, columnIndexClass: Int) extends FileReader(path) {

  def read():(DenseMatrix[Double],DenseVector[Int])={

    var maxColumns = 0
    var numLines = 0

    //First, check the number of lines and columns of the data set
    val bufferedSource: BufferedSource = scala.io.Source.fromFile(path)
    for ((line, count) <- bufferedSource.getLines.zipWithIndex) {
      val cols = line.split(separator).map(_.trim).toList
      maxColumns = max(maxColumns, cols.size)
      numLines = numLines + 1
    }
    bufferedSource.close

    println("The input file " + path + " has "+ numLines +" lines and "+ maxColumns +" number of columns.")

    val X: DenseMatrix[Double] = DenseMatrix.zeros[Double](numLines, maxColumns - 1)
    val Y: DenseVector[Int] = DenseVector.zeros[Int](numLines)

    //Second, read in the whole data set and store the data into separate matrices
    val bufferedSource2: BufferedSource = scala.io.Source.fromFile(path)
    for ((line, count) <- bufferedSource2.getLines.zipWithIndex) {
      val array : Array[Double] = line.split(separator).map(_.trim).map(_.toDouble)
      for(column <- 0 until maxColumns){
        var yHasPassed = false
        if(column == columnIndexClass){
          Y(count)=array(column).toInt
          yHasPassed = true
        } else{
          val index : Int = if(yHasPassed) column-1 else column
          X(count, index) = array(column)
        }
      }
    }
    bufferedSource2.close
    (X,Y)
  }
}