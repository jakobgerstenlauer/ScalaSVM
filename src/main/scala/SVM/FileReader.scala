package SVM
import breeze.linalg._

import scala.io.{BufferedSource, Source}
abstract class FileReader(path: String)

/**
  *
  * @param path The path to the input file.
  * @param separator The column separator in the csv file.
  * @param columnIndexClass The index of the column (starting with 0) containing the labels.
  */
class CSVReader(path: String, separator: Char, columnIndexClass: Int) extends FileReader(path) {
  /**
    * Reads a csv data set with the given separator and returns the data matrix and the reponse vector.
    * @return (X,Y) A tuple with the input matrix and the response vector.
    */
  def read(): (DenseMatrix[Double], DenseVector[Int]) ={

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