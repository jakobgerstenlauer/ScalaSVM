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


  /**
    * Measures the processing time of a given Scala command.
    * Source: http://biercoff.com/easily-measuring-code-execution-time-in-scala/
    * @param block The code block to execute.
    * @tparam R
    * @return
    */
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

	val kernelPar = GaussianKernelParameter(1.0)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
	val N = 100000
  Utility.testJVMArgs(N/2)
	val dataProperties = DataParams(N = N, d = 10, ratioTrain = 0.5)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)

  //First find a value for epsilon that is manageable:
	//val probeMatrices = ProbeMatrices(d, gaussianKernel)

	//Number of non-sparse matrix elements with epsilon = 0.001:
	val epsilon = 0.001
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
	val ap = AlgoParams(maxIter = 30, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.5, epsilon = epsilon, isDebug = false, hasMomentum = false, quantileAlphaClipping=0.03)
	var algo = new NoMatrices(alphas, ap, mp, lmf)
	var numInt = 0
  while(numInt < ap.maxIter && algo.getSparsity() < 99.0){
		algo = algo.iterate()
		numInt += 1
	}
 /* Synthetic dataset with 10 variables.
  Observations: 50000 (training), 50000(test)
  Data was already generated.

  Elapsed time: 563596072ns
  Sequential approach: The matrix has 50000 rows.
    The hash map has 49960 <key,value> pairs.
    Elapsed time: 425812077187ns
  The hash map has 49970 <key,value> pairs.
    Elapsed time: 1179510652053ns
    Train:42511/50000=85%,Test:38139/50000=76%,Sparsity:0%
Train:44358/50000=89%,Test:37603/50000=75%,Sparsity:7%
Train:44861/50000=90%,Test:36681/50000=73%,Sparsity:10%
Train:44990/50000=90%,Test:36286/50000=73%,Sparsity:11%
Train:45093/50000=90%,Test:35826/50000=72%,Sparsity:12%
Train:45037/50000=90%,Test:35372/50000=71%,Sparsity:13%
Train:44945/50000=90%,Test:35024/50000=70%,Sparsity:14%
Train:44734/50000=89%,Test:34535/50000=69%,Sparsity:15%
Train:44643/50000=89%,Test:34309/50000=69%,Sparsity:16%
Train:44521/50000=89%,Test:33982/50000=68%,Sparsity:17%
Train:44045/50000=88%,Test:33566/50000=67%,Sparsity:18%
Train:43632/50000=87%,Test:33340/50000=67%,Sparsity:18%
  */
}