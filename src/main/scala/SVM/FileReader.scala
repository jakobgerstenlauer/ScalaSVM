package SVM
import java.io.IOException

import breeze.linalg._

import scala.io.{BufferedSource, Source}
abstract class FileReader(path: String)

/**
  *
  * @param path The path to the input file.
  * @param separator The column separator in the csv file.
  * @param columnIndexClass The index of the column (starting with 0) containing the labels.
  * @param columnIndexIgnore The index of a column (starting with 0) which should be ignored (i.e. a line nr).
  */
class CSVReader(path: String, separator: Char, columnIndexClass: Int, columnIndexIgnore: Int = -1) extends FileReader(path) {
  /**
    * Reads a csv data set with the given separator and returns the data matrix and the reponse vector.
    * @return (X,Y) A tuple with the input matrix and the response vector.
    */
  def read(transformLabel: Double => Int = (x:Double)=>if(x>0) 1 else -1): (DenseMatrix[Double], DenseVector[Int]) = {
    var maxColumns = 0
    var numLines = 0
    try {
      //First, check the number of lines and columns of the data set
      val bufferedSource: BufferedSource = scala.io.Source.fromFile(path)
      for ((line, _) <- bufferedSource.getLines.zipWithIndex) {
        val cols = line.split(separator).map(_.trim).toList
        maxColumns = max(maxColumns, cols.size)
        numLines = numLines + 1
      }
      bufferedSource.close
    }catch {
      case e: IOException => e.printStackTrace(); e.toString
    }
    println("The input file " + path + " has "+ numLines +" lines and "+ maxColumns +" columns.")
    val hasColumnToIgnore : Int = if (columnIndexIgnore >= 0) 1 else 0
    val X: DenseMatrix[Double] = DenseMatrix.zeros[Double](numLines, maxColumns - 1 - hasColumnToIgnore)
    val Y: DenseVector[Int] = DenseVector.zeros[Int](numLines)
    //Second, read in the whole data set and store the data into separate matrices
    try {
      val bufferedSource2: BufferedSource = scala.io.Source.fromFile(path)
      for ((line, count) <- bufferedSource2.getLines.zipWithIndex) {
        val array : Array[Double] = line.split(separator).map(_.trim).map(_.toDouble)
        var hasPassed = 0
        for(column <- 0 until maxColumns){
          column match {
            case `columnIndexIgnore` => hasPassed = hasPassed + 1
            case `columnIndexClass` =>
              Y(count)=transformLabel(array(column).toInt)
              hasPassed = hasPassed + 1
            case _ =>
              val index : Int = column - hasPassed
              X(count, index) = array(column)
          }
        }
      }
      bufferedSource2.close
    }
    catch {
      case e: IOException => e.printStackTrace(); e.toString
    }
    (X,Y)
  }
}