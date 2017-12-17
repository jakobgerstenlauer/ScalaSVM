package SVM

object TestLocalData extends App {

  val dataProperties = DataParams(N=19020, d=10, ratioTrain=0.5)
  println(dataProperties)

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathTest = workingDir + "magic04test.csv"

  d.readTrainingDataSet (pathTrain, ',', 10)
  d.readTestDataSet (pathTest, ',', 10)
  println(d)
}