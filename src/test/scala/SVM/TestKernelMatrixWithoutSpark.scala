package test
import java.util

import SVM._
import scala.collection.JavaConverters._
import scala.util.matching.Regex
// import what we need
import java.lang.management.ManagementFactory
import java.lang.management.RuntimeMXBean
import SVM.DataSetType.{Test, Train}

//Important flags for the Java virtual machine:
//Force the JVM to cache Integers up to dimensionality of K and S:
//-Djava.lang.Integer.IntegerCache.high=50000
//This way duplicate integers in the HashMaps are cached
// and memory footprint is significantly reduced! (Flyweight pattern)
//Also define the maximum available RAM:
// -Xmx14G
//And the minimum available RAM (should be close to or equal to max):
// -Xms8G
object testKernelMatrixWithoutSpark extends App {
  //http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

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

	val kernelPar = GaussianKernelParameter(1.0)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
	val N = 100000
  testJVMArgs(N/2)
	val dataProperties = DataParams(N = N, d = 10, ratioTrain = 0.5)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)

  //First find a value for epsilon that is manageable:
	//val probeMatrices = ProbeMatrices(d, gaussianKernel)

	//Number of non-sparse matrix elements with epsilon = 0.001:
	val epsilon = 0.01
	//val numElementsS =  probeMatrices.probeSparsity(Test, 0.001)
	//val numElementsK =  probeMatrices.probeSparsity(Train, 0.001)
  //println("Projected memory requirements for epsilon ="+epsilon+":")
  //Integer = 32 bits = 4 Byte
  //val intsPerKB = 256
  //println("Training matrix K: "+numElementsK/intsPerKB+"kB:")
  //println("Training matrix S: "+numElementsS/intsPerKB+"kB:")

  val lmf = time{LeanMatrixFactory(d, gaussianKernel, epsilon)}
	val mp = ModelParams(C = 0.5, delta = 0.1)
	val alphas = new Alphas(N=N/2, mp)
	val ap = AlgoParams(maxIter = 30, batchProb = 0.99, minDeltaAlpha = 0.001, learningRateDecline = 0.5, epsilon = epsilon, isDebug = false, hasMomentum = false, quantileAlphaClipping=0.03)
	var algo = new NoMatrices(alphas, ap, mp, lmf)
	var numInt = 0
  while(numInt < ap.maxIter && algo.getSparsity() < 99.0){
		algo = algo.iterate()
		numInt += 1
	}
}