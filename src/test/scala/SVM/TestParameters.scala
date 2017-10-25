package test
import SVM.ModelParams
import SVM.DataParams
import SVM.Data

object testParameters extends App {
	val dataProperties = DataParams(N=100, d=10, ratioTrain=0.5)
	val modelProperties = ModelParams(sigma=1.0, epsilon=0.0001, C=1.0, lambda=0.5, batchProb=0.1) 
	val d = new Data(dataProperties)
	d.simulate()
	println(d.z_train)
}
