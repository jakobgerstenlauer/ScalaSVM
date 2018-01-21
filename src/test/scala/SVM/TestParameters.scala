package test
import SVM.ModelParams
import SVM.DataParams
import SVM.SimData

object TestParameters extends App {
	val dataProperties = DataParams(N=100, d=10)
	println(dataProperties)
	val modelProperties = ModelParams()
	println(modelProperties) 
	val d = new SimData(dataProperties)
	d.simulate()
	println(d)
}
