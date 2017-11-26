package test
import SVM.Alphas
import breeze.linalg._
import breeze.numerics._

object testAlphas extends App {
	val alpha = new DenseVector(Array(1.0,1.0))
	val alpha_old = new DenseVector(Array(-1.0,1.0))
	val alphas = Alphas(2, alpha, alpha_old)
	println("alpha_old: "+alphas.alphaOld)
	println("alpha before update : "+alphas.alpha)
	alphas.updateAlphaAsConjugateGradient()
	println("alpha after update: "+alphas.alpha)
}
