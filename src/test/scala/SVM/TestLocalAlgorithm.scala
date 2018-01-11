package SVM

object TestLocalAlgorithm extends App {

  val d = new LocalData()
  println(d)

  val workingDir = "/home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/"
  val pathTrain = workingDir + "magic04train.csv"
  val pathTest = workingDir + "magic04test.csv"

  d.readTrainingDataSet (pathTrain, ',', 11)
  d.readTestDataSet (pathTest, ',', 11)
  d.tableLabels()

  val epsilon = 0.0001
  val kernelPar = GaussianKernelParameter(1.0)
  val gaussianKernel = GaussianKernel(kernelPar)
  val kmf = LeanMatrixFactory(d, gaussianKernel, epsilon)
  val mp = ModelParams(C = 0.5, delta = 0.01)
  val alphas = new Alphas(N=d.N_train, mp)
  val ap = AlgoParams(maxIter = 20, batchProb = 0.9, minDeltaAlpha = 0.001, learningRateDecline = 0.5,
    epsilon = epsilon, isDebug = false, hasMomentum = false, quantileAlphaClipping = 0.0)
  var algo1 = NoMatrices(alphas, ap, mp, kmf)
  var numInt = 0
  while(numInt < ap.maxIter){
    algo1 = algo1.iterate()
    numInt += 1
  }
}
/*
/usr/lib/jvm/java-8-oracle/bin/java -Xmx14G -Xms14G -Djava.lang.Integer.IntegerCache.high=1000000
Empirical dataset from local file system with 0 variables.
Observations: 0 (training), 0 (test)
Data was not yet generated.

The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04train.csv has 12680 lines and 12 columns.
Summary statistics of train data set before z-transformation:
mean:		variance:		standard deviation:
6339.5               1.339959E7            3660.545041383865
53.272122673501556   1818.0896606786725    42.63906261491536
22.11556955835961    339.1220152777037     18.41526582153252
2.8227191640378555   0.22337316786206698   0.47262370641141876
0.3809673580441642   0.03326486575435714   0.18238658326301618
0.21497101735015878  0.012109416777259746  0.11004279520831768
-4.8474820504732     3563.4296437166977    59.69446912165898
10.030569227129366   2636.1222805503703    51.34318144165173
0.27793725552050397  432.2682987920789     20.791062954839006
27.756945457413263   679.4403600041117     26.066076804999092
193.26211767350148   5561.9388184949885    74.57840718663137
Summary statistics of train data set AFTER z-transformation:
mean:	variance:	standard deviation:
-2.600160755109049E-14   1.0000000000001201  1.00000000000006
5.232190277825594E-16    0.999999999999996   0.999999999999998
6.369518356762888E-16    0.9999999999999987  0.9999999999999993
-1.2910510063367936E-15  0.9999999999999936  0.9999999999999968
-9.292883167621654E-16   1.0000000000000056  1.0000000000000027
-9.535524225692121E-15   0.999999999999997   0.9999999999999984
2.3889039618002483E-16   1.0000000000000016  1.0000000000000007
-5.493042544262028E-16   0.9999999999999964  0.9999999999999982
3.2912312198513094E-17   0.9999999999999986  0.9999999999999992
-5.391906810359864E-16   1.0000000000000036  1.0000000000000018
1.3268737237420725E-15   1.000000000000003   1.0000000000000016
The input file /home/jakob/workspace_scala/Dist_Online_SVM/data/MagicGamma/magic04test.csv has 6340 lines and 12 columns.
Summary statistics of test data set BEFORE z-transformation with means and standard deviation of the training set:
mean:	variance:	standard deviation:
3169.5               3350161.6666666665    1830.3446852073155
53.2062164353311     1748.4400415036146    41.81435209953174
22.311759542586692   331.5163448605896     18.20759030900546
2.829612555205062    0.22330565872544658   0.4725522814731155
0.37904649842271343  0.033735045494914885  0.1836710251915497
0.21402936908517417  0.012420428110344925  0.11144697443333725
-3.3002713722397643  3388.161702489047     58.20791786766683
11.575495993690856   2529.6048812468284    50.295177514815755
0.1933033596214506   436.8739352261841     20.901529494900224
27.423229116719217   685.3498372498752     26.179187100631587
194.9298440536277    5629.672452385667     75.03114321657152
Summary statistics of test data set AFTER z-transformation with means and standard deviation of the training set:
mean:	variance:	standard deviation:
-0.8659912565375991     0.2500197145336576  0.5000197141450101
-0.0015456774640085985  0.9616907676879619  0.9806583338186455
0.010653660182181082    0.9775724663263593  0.9887226437815406
0.01458536902334537     0.9996977741898604  0.9998488756756495
-0.010531803310779872   1.0141344247119384  1.0070424145545898
-0.008557109651773908   1.0256834279309952  1.0127603013206012
0.025918828008675277    0.9508148164124183  0.9750973368912553
0.03009020327883636     0.9595931493430994  0.9795882550046725
-0.004070686336859687   1.0106545782954117  1.005313174237467
-0.012802706874170629   1.008697565810967   1.0043393678488197
0.022362054152656026    1.0121780616617813  1.0060706047101173
Training: 8173(+)/4507(-)/12680(total)
Test: 4159(+)/2181(-)/6340(total)
Elapsed time: 98593569ns
Sequential approach: The matrix has 12680 rows.
The hash map has 12639 <key,value> pairs.
The matrix has 46102494 non-sparse elements.
Elapsed time: 39405593187ns
The hash map has 12572 <key,value> pairs.
The matrix has 22846588 non-sparse elements.
In z with length: 12680 there are: 8173 positive and: 4507 negative entries!
In z with length: 6340 there are: 5171 positive and: 1142 negative entries!
Train:10406/12680=82%,Test:5070/6340=80%,Sparsity:0%
Jan 09, 2018 5:30:06 PM com.github.fommil.jni.JniLoader liberalLoad
INFO: successfully loaded /tmp/jniloader8319969539016792807netlib-native_system-linux-x86_64.so
In z with length: 12680 there are: 6839 positive and: 4497 negative entries!
In z with length: 6340 there are: 4729 positive and: 1584 negative entries!
Train:10736/12680=85%,Test:5196/6340=82%,Sparsity:11%
In z with length: 12680 there are: 6507 positive and: 4466 negative entries!
In z with length: 6340 there are: 4700 positive and: 1613 negative entries!
Train:10815/12680=85%,Test:5213/6340=82%,Sparsity:13%
In z with length: 12680 there are: 6308 positive and: 4409 negative entries!
In z with length: 6340 there are: 4768 positive and: 1545 negative entries!
Train:10894/12680=86%,Test:5213/6340=82%,Sparsity:15%
In z with length: 12680 there are: 6142 positive and: 4351 negative entries!
In z with length: 6340 there are: 4800 positive and: 1513 negative entries!
Train:10943/12680=86%,Test:5233/6340=83%,Sparsity:17%
In z with length: 12680 there are: 5998 positive and: 4288 negative entries!
In z with length: 6340 there are: 4849 positive and: 1464 negative entries!
Train:10968/12680=86%,Test:5244/6340=83%,Sparsity:19%
In z with length: 12680 there are: 5854 positive and: 4237 negative entries!
In z with length: 6340 there are: 4890 positive and: 1423 negative entries!
Train:10978/12680=87%,Test:5259/6340=83%,Sparsity:20%
In z with length: 12680 there are: 5714 positive and: 4165 negative entries!
In z with length: 6340 there are: 4920 positive and: 1392 negative entries!
Train:10961/12680=86%,Test:5250/6340=83%,Sparsity:22%
In z with length: 12680 there are: 5629 positive and: 4117 negative entries!
In z with length: 6340 there are: 4948 positive and: 1364 negative entries!
Train:10961/12680=86%,Test:5248/6340=83%,Sparsity:23%
In z with length: 12680 there are: 5517 positive and: 4059 negative entries!
In z with length: 6340 there are: 4968 positive and: 1343 negative entries!
Train:10964/12680=86%,Test:5239/6340=83%,Sparsity:24%
In z with length: 12680 there are: 5443 positive and: 4006 negative entries!
In z with length: 6340 there are: 4919 positive and: 1392 negative entries!
Train:10956/12680=86%,Test:5248/6340=83%,Sparsity:25%
In z with length: 12680 there are: 5468 positive and: 3955 negative entries!
In z with length: 6340 there are: 4907 positive and: 1404 negative entries!
Train:10919/12680=86%,Test:5212/6340=82%,Sparsity:26%
In z with length: 12680 there are: 5376 positive and: 3930 negative entries!
In z with length: 6340 there are: 4872 positive and: 1439 negative entries!
Train:10950/12680=86%,Test:5223/6340=82%,Sparsity:27%
In z with length: 12680 there are: 5343 positive and: 3886 negative entries!
In z with length: 6340 there are: 4877 positive and: 1434 negative entries!
Train:10963/12680=86%,Test:5220/6340=82%,Sparsity:27%
In z with length: 12680 there are: 5302 positive and: 3832 negative entries!
In z with length: 6340 there are: 4854 positive and: 1457 negative entries!
Train:10936/12680=86%,Test:5205/6340=82%,Sparsity:28%
In z with length: 12680 there are: 5277 positive and: 3778 negative entries!
In z with length: 6340 there are: 4915 positive and: 1396 negative entries!
Train:10943/12680=86%,Test:5196/6340=82%,Sparsity:29%
In z with length: 12680 there are: 5208 positive and: 3738 negative entries!
In z with length: 6340 there are: 4932 positive and: 1379 negative entries!
Train:10889/12680=86%,Test:5189/6340=82%,Sparsity:29%
In z with length: 12680 there are: 5261 positive and: 3682 negative entries!
In z with length: 6340 there are: 5042 positive and: 1268 negative entries!
Train:10804/12680=85%,Test:5168/6340=82%,Sparsity:29%
In z with length: 12680 there are: 5146 positive and: 3651 negative entries!
In z with length: 6340 there are: 5032 positive and: 1278 negative entries!
Train:10722/12680=85%,Test:5096/6340=80%,Sparsity:31%
In z with length: 12680 there are: 5150 positive and: 3639 negative entries!
In z with length: 6340 there are: 5062 positive and: 1248 negative entries!
Train:10755/12680=85%,Test:5110/6340=81%,Sparsity:31%

Process finished with exit code 0
*/
