package test
import SVM.ModelParams
import SVM.DataParams
import SVM.Data

object testParameters extends App {
	val dataProperties = DataParams(N=100, d=10, ratioTrain=0.5)
	println(dataProperties)
	val modelProperties = ModelParams(C=1.0, lambda=0.5)
	println(modelProperties) 
	val d = new Data(dataProperties)
	d.simulate()
	println(d)
}
