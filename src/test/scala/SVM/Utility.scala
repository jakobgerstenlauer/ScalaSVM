package SVM

import java.lang.management.ManagementFactory
import scala.collection.JavaConverters._
import scala.util.matching.Regex

object Utility {

  /**
    * Tests if the parameter java.lang.Integer.IntegerCache.high is correctly set.
    *
    * The Java Virtual Machine is able to cache Integer in order to implement
    * Flightweight pattern and reduce the memmory footprint of Integer objects.
    * In order to use this strategy, it is necessary to set the value of the parameter
    * "java.lang.Integer.IntegerCache.high" to the dimensionality of the kernel matrices.
    * @param intMax
    */
  def testJVMArgs(intMax: Int):Unit= {

    val integerCacheMax : Regex = new Regex("-Djava.lang.Integer.IntegerCache.high=\\d+")
    val cache : Regex = new Regex("\\d+")

    // get a RuntimeMXBean reference
    val runtimeMxBean = ManagementFactory.getRuntimeMXBean

    // get the jvm's input arguments as a list of strings
    val listOfArguments = runtimeMxBean.getInputArguments.asScala

    for (arg <- listOfArguments){
      val matchingArgument : Option[String] = integerCacheMax findFirstIn arg
      if(matchingArgument.isDefined){
        println("The argument IntegerCache.high is defined as "+ matchingArgument.get.toString())
        val valueMaxInt : Int = (cache findFirstIn matchingArgument.get).get.toString().toInt
        if(valueMaxInt < intMax) println("Warning!!! The argument IntegerCache.high should be increased to: "+ intMax +" to decrease memory.")
      }
    }
  }
}
