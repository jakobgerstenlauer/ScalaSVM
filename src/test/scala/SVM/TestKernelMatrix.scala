package test
import SVM._
import org.apache.spark.{SparkConf, SparkContext}

object TestKernelMatrix extends App {
	val kernelPar = GaussianKernelParameter(1.5)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
 	val dataProperties = DataParams(N=100, d=10)
	println(dataProperties)
	val d = new SimData(dataProperties)
	println(d)
	d.simulate()
	println(d)
	val appName = "TestKernelMatrix"
	val conf = new SparkConf().setAppName(appName).setMaster("spark://jakob-Lenovo-G50-80:7077")
	val sparkContext = new SparkContext(conf)
	val epsilon = 0.0001 
	val fac = KernelMatrixFactory(d, gaussianKernel, epsilon, sparkContext)
}
