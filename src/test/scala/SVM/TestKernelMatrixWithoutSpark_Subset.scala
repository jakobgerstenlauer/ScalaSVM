import SVM._
import SVM.DataSetType.{Test, Validation}
import scala.collection.mutable.ListBuffer
import scala.concurrent.{Await, Future}

//Important flags for the Java virtual machine:
//Force the JVM to cache Integers up to dimensionality of K and S:
//-Djava.lang.Integer.IntegerCache.high=50000
//This way duplicate integers in the HashMaps are cached
// and memory footprint is significantly reduced! (Flyweight pattern)
//All flags:
//-server -XX:+UseConcMarkSweepGC -XX:+CMSIncrementalMode -XX:+CMSIncrementalPacing -XX:CMSIncrementalDutyCycleMin=0 -XX:CMSIncrementalDutyCycle=10 -XX:+UseCMSInitiatingOccupancyOnly -XX:ThreadStackSize=300 -XX:MaxTenuringThreshold=0 -XX:SurvivorRatio=128 -XX:+UseTLAB -XX:+PrintGCDetails -Xms12288M  -Xmx12288M  -XX:NewSize=3072M  -XX:MaxNewSize=3072M -XX:ParallelGCThreads=4 -Djava.lang.Integer.IntegerCache.high=1000000 -verbose:gc -Xloggc:"/home/jakob/Documents/UPC/master_thesis/jvm/logs"
object TestKernelMatrixWithoutSpark_Subset extends App {
	/*** Measures the processing time of a given Scala command.
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
	val gaussianKernel = GaussianKernel(GaussianKernelParameter(1.0))
	println(gaussianKernel)
	val N = 300000
	//Utility.testJVMArgs(N/2)
	val dataProperties = DataParams(N = N, d = 10)
	println(dataProperties)
	val d = new SimData(dataProperties)
	//println(d)
	d.simulate()
	println(d)

	val N_train = d.selectInstances()

	//First find a value for epsilon that is manageable:
	//Number of non-sparse matrix elements with epsilon = 0.001:
	val epsilon = 0.001
	//probeSparsity(epsilon: Double, typeOfMatrix: DataSetType.Value,kf: KernelFunction)
	val sampleProb = 0.01
	//val ratioNonSparseElementsTrain =  d.probeSparsity(epsilon, Train, gaussianKernel,sampleProb)
	//val ratioNonSparseElementsValidation =  d.probeSparsity(epsilon, Validation, gaussianKernel,sampleProb)
	//val ratioNonSparseElementsTest =  d.probeSparsity(epsilon, Test, gaussianKernel,sampleProb)
	//println("Projected memory requirements for epsilon ="+epsilon+":")
	//Integer = 32 bits = 4 Byte
	//There is also some overhead in the hash map, so I assume that one matrix element takes up to 2 Ints:
	//val intsPerKB = 256 * 0.5
	//val N_train = N * 0.5
	//val N_val = N * 0.4
	//val N_test = N * 0.1
	//println("Training matrix: "+ (ratioNonSparseElementsTrain * N_train*N_train)/intsPerKB+"kB:")
	//println("Validation matrix: "+ (ratioNonSparseElementsValidation * N_val*N_val)/intsPerKB+"kB:")
	//println("Test matrix: "+(ratioNonSparseElementsTrain * N_test*N_test)/intsPerKB+"kB:")

	val lmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
	val mp = ModelParams(C = 0.4, delta = 0.01)
	val alphas = new Alphas(N=N_train, mp)
	val ap = AlgoParams(batchProb = 0.99, learningRateDecline = 0.8, epsilon = epsilon)
	var algo = NoMatrices(alphas, ap, mp, lmf, new ListBuffer[Future[(Int,Int,Int)]])
	var numInt = 0
	while(numInt < ap.maxIter && algo.getSparsity < 99.0){
		algo = algo.iterate(numInt)
		numInt += 1
	}

	val future = algo.predictOn(Validation, PredictionMethod.AUC)
	Await.result(future, LeanMatrixFactory.maxDuration)

	val future2 = algo.predictOn(Test, PredictionMethod.THRESHOLD, 0.49)
	Await.result(future2, LeanMatrixFactory.maxDuration)
}