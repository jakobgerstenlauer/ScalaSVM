package SVM

object TestLocalAlgorithm extends App {

  val dataProperties = DataParams(N=19020, d=10, ratioTrain=0.5)
  println(dataProperties)

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathTest = workingDir + "magic04test.csv"

  d.readTrainingDataSet (pathTrain, ',', 11)
  d.readTestDataSet (pathTest, ',', 11)
  println(d)

  val epsilon = 0.00001
  val kernelPar = GaussianKernelParameter(10000.0)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 1.0, delta = 0.1)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter = 10, batchProb = 0.8, minDeltaAlpha = 0.001, learningRateDecline = 0.5,
    epsilon = epsilon, isDebug = false, hasMomentum = false)
  var algo1 = NoMatrices(alphas, ap, mp, kmf)
  var numInt = 0
  while(numInt < 20){
    algo1 = algo1.iterate()
    numInt += 1
  }
}
/*
Data parameters:
Total number of observations: 19020
Observations training set: 9510
Observations test set: 9510
Number of features: 10

Empirical dataset from local file system with 0 variables.
Observations: 0 (training), 0(test)
Data was not yet generated.

The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04train.csv has 12680 lines and 12 columns.
The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04test.csv has 6340 lines and 12 columns.
Empirical dataset from local file system with 11 variables.
Observations: 12680 (training), 6340(test)
Data was already generated.

Elapsed time: 109891177ns
Sequential approach: The matrix has 12680 rows.
The hash map has 0 <key,value> pairs.
The matrix has 12680 non-sparse elements.
Elapsed time: 48116269775ns
The hash map has 0 <key,value> pairs.
The matrix has 0 non-sparse elements.
Train:12680/12680=100%,Test:0/6340=0%,Sparsity:0%
Jan 08, 2018 5:37:55 PM com.github.fommil.jni.JniLoader liberalLoad
INFO: successfully loaded /tmp/jniloader5256028516163924169netlib-native_system-linux-x86_64.so
Train:8058/12680=64%,Test:0/6340=0%,Sparsity:36%
Train:6436/12680=51%,Test:0/6340=0%,Sparsity:49%
Train:5519/12680=44%,Test:0/6340=0%,Sparsity:56%
Train:4970/12680=39%,Test:0/6340=0%,Sparsity:61%
Train:4602/12680=36%,Test:0/6340=0%,Sparsity:64%
Train:4351/12680=34%,Test:0/6340=0%,Sparsity:66%
Train:4163/12680=33%,Test:0/6340=0%,Sparsity:67%
Train:4024/12680=32%,Test:0/6340=0%,Sparsity:68%
Train:3915/12680=31%,Test:0/6340=0%,Sparsity:69%
Train:3806/12680=30%,Test:0/6340=0%,Sparsity:70%
Train:3720/12680=29%,Test:0/6340=0%,Sparsity:71%
Train:3661/12680=29%,Test:0/6340=0%,Sparsity:71%
Train:3616/12680=29%,Test:0/6340=0%,Sparsity:71%
Train:3594/12680=28%,Test:0/6340=0%,Sparsity:72%
Train:3587/12680=28%,Test:0/6340=0%,Sparsity:72%
Train:3576/12680=28%,Test:0/6340=0%,Sparsity:72%
Train:10864/12680=86%,Test:0/6340=0%,Sparsity:14%
Train:12318/12680=97%,Test:0/6340=0%,Sparsity:3%
Train:12612/12680=99%,Test:0/6340=0%,Sparsity:1%

Process finished with exit code 0*/
