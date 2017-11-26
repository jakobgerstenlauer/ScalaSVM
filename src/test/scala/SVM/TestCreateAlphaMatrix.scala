package test

import SVM.Alphas
import SVM.Algorithm
import SVM.Parameters
import SVM.AlgoParams
import SVM.ModelParams
import SVM.DataParams
import SVM.Data
import SVM.KernelMatrixFactory
import SVM.DistributedMatrixOps
import SVM.GaussianKernelParameter
import SVM.GaussianKernel
import SVM.hasBagging
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import breeze.linalg._
import breeze.numerics._

/**
*Stochastic gradient descent algorithm
**/
class TestClassSGD(var alphas: Alphas, val ap: AlgoParams, val mp: ModelParams, val kmf: KernelMatrixFactory, sc: SparkContext) 
        extends Algorithm with hasBagging{

        val matOps : DistributedMatrixOps = new DistributedMatrixOps(sc)

        def iterate() : Unit = {

                //Decrease the step size, i.e. learning rate:
                mp.updateDelta(ap)

		println("alphas:" + alphas.alpha)

                //Create a random sample of alphas and store it in a distributed matrix Alpha:
                val Alpha = getDistributedAlphas(ap, alphas, kmf, sc)

		matOps.printFirstRow(Alpha)
        }
}



object testCreateAlphaMatrix extends App {
	val N = 100
	val kernelPar = GaussianKernelParameter(1.5)
	val gaussianKernel = GaussianKernel(kernelPar)
	val dataProperties = DataParams(N=N, d=5, ratioTrain=0.5)
	val d = new Data(dataProperties)
	d.simulate()
	val epsilon = 0.001
	val appName = "TestCreateAlphaMatrix"
        val conf = new SparkConf().setAppName(appName).setMaster("spark://jakob-Lenovo-G50-80:7077")
        val sc = new SparkContext(conf)
	//In Databricks notebooks and Spark REPL, the SparkSession has been created automatically and assigned to variable "spark".
	//SparkSession.sparkContext returns the underlying SparkContext, used for creating RDDs as well as managing cluster resources.
	//Compare: https://docs.databricks.com/spark/latest/gentle-introduction/sparksession.html
	//val sc = spark.sparkContext
	val kmf = new KernelMatrixFactory(d, gaussianKernel, epsilon, sc)
	val alphas = new Alphas(N=(0.5*N).toInt)
	val ap = AlgoParams(maxIter = 30, minDeltaAlpha = 0.001, learningRateDecline = 0.5,
	numBaggingReplicates = 100, batchProb = 0.1, epsilon = 0.0001, isDebug = false)
	val mp = ModelParams(C = 1.0, lambda = 0.1)
	val algo1 = new TestClassSGD(alphas, ap, mp, kmf, sc)
	var numInt = 0
	algo1.iterate()
}

