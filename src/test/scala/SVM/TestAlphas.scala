package test
import SVM.Alphas
import breeze.linalg._
import breeze.numerics._

object testAlphas extends App {
	val N = 1000
	val alpha : DenseVector[Double] = DenseVector.rand(N)
	val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
	val alphas = Alphas(N, alpha, alpha_old)
	println("alphas: "+alphas.alpha)
	println("10% Quantile: "+alphas.getQuantile(0.1))
	println("20% Quantile: "+alphas.getQuantile(0.2))
	println("Median: "+alphas.getQuantile(0.5))
	println("90% Quantile: "+alphas.getQuantile(0.9))
	//println("alpha before update : "+alphas.alpha)
	//alphas.updateAlphaAsConjugateGradient()
	//println("alpha after update: "+alphas.alpha)
}
