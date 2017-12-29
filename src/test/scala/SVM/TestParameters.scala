package test
import SVM.ModelParams
import SVM.DataParams
import SVM.SimData

object testParameters extends App {
	val dataProperties = DataParams(N=100, d=10, ratioTrain=0.5)
	println(dataProperties)
	val modelProperties = ModelParams(C=1.0)
	println(modelProperties) 
	val d = new SimData(dataProperties)
	d.simulate()
	println(d)
}
