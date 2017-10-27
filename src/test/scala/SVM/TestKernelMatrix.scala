package test
import SVM.KernelParameter
import SVM.GaussianKernelParameter
import SVM.KernelFunction
import SVM.GaussianKernel
import SVM.KernelMatrixFactory
import SVM.DataParams
import SVM.Data
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object testKernelMatrix extends App {
	val kernelPar = GaussianKernelParameter(1.5)
	println(kernelPar)
	val gaussianKernel = GaussianKernel(kernelPar)
	println(gaussianKernel)
 	val dataProperties = DataParams(N=100, d=10, ratioTrain=0.5)
	println(dataProperties)
        val d = new Data(dataProperties)
	println(d)
        d.simulate()
	println(d)
	val appName = "TestKernelMatrix"
	//TODO Define name of master and start actual spark cluster.
	val master = "master"
	val conf = new SparkConf().setAppName(appName).setMaster(master)
	val sparkContext = new SparkContext(conf)
	val epsilon = 0.0001 
	val fac = new KernelMatrixFactory(d, gaussianKernel, epsilon, sparkContext)
	val K = fac.getKernelMatrixTraining()
	val S = fac.getKernelMatrixTest()
}
