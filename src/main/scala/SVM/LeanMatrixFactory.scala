package SVM

import breeze.linalg.{DenseVector, _}
import breeze.numerics.signum

import scala.collection.mutable
import scala.collection.mutable.{HashMap, MultiMap, Set => MSet}
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.{Failure, Success}
import scala.util.Random

case class LeanMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){

  /**
    * Cached diagonal of K, i.e. the kernel matrix of the training set.
    */
  val diagonal : DenseVector[Double]= initializeDiagonal()

  def initializeDiagonal():DenseVector[Double]={
    val N = d.getN_train
    val diagonal = DenseVector.zeros[Double](N)
    for (i <- 0 until N){
      diagonal(i) = kf.kernel(d.getRowTrain(i), d.getRowTrain(i))
    }
    diagonal
  }

  //Measures execution time of a block of code and prints it to std out.
  //Usage: time{code}
  //http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

  /**
    * key: row index of matrix K
    * value: set of non-sparse column indices of matrix K
    */
  val rowColumnPairs : MultiMap[Integer, Integer] = initializeRowColumnPairs4Threads();

  /**
    * key: row index of matrix K
    * value: set of non-sparse column indices of matrix K
    */
  //val rowColumnPairsParallel : MultiMap[Integer, Integer] = initializeRowColumnPairs2();

  /**
    * Check if the similarity between instance i and j is significant.
    * @param i row index of K
    * @param j column index of K
    * @return
    */
  def entryExists(i: Integer, j: Integer):Boolean = {
    rowColumnPairs.entryExists(i, _ == j)
  }

  /**
    * Check if the similarity between test instance i and training instance j is significant.
    * @param i row index of S
    * @param j column index of S
    * @return
    */
  def entryExistsTest(i: Integer, j: Integer):Boolean = {
    rowColumnPairsTest.entryExists(i, _ == j)
  }

  /**
    * key: row index of matrix S (index of test instance)
    * value: set of non-sparse column indices of matrix S (index of trainings instance)
    */
  val rowColumnPairsTest : MultiMap[Integer, Integer] = initializeRowColumnPairsTest(true);

 /* /**
    * Parallel version that works on stream.
    * Is not used because the performance is not better and synchronization issues are not solved.
    * @return
    */
  def initializeRowColumnPairs2(): MultiMap[Integer , Integer] = {
    //FIXME: This MultiMap implementation is build on top of HashMap which is probably not threadsafe!
    val mmap: MultiMap[Integer, Integer] = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]

    val N = d.getN_train
    //val NumElements = N*N

   //add the diagonal by default without calculating the kernel function
    for (i <- 0 until N){
      mmap.addBinding(i,i)
    }

    val localFunction = (ind: Indices) => {
      mmap.addBinding(ind.i, ind.j)
      mmap.addBinding(ind.j, ind.i)
    }

    val filteredParallelStream =
    MatrixIndexStream.getMatrixIndexStream(N)
      .par
      .filter(x => kf.kernel(d.getRowTrain(x.i), d.getRowTrain(x.j)) > epsilon)
      //.map(x => localFunction(x))


    filteredParallelStream.toArray.foreach(localFunction)
    mmap
  }*/

  /**
    *
    * @param isCountingSparsity Should the sparsity of the matrix representation be assessed? Involves some overhead.
    * @return
    */
  def initializeRowColumnPairs(isCountingSparsity: Boolean): MultiMap[Integer, Integer] = {
    println("Preparing the hash map for the training set.")
    val map: MultiMap[Integer, Integer] = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
    val N = d.getN_train
    var size2 : Int = N
    val maxIterations : Int = (N * N - N) / 2
    println("Number of iterations: "+ maxIterations)
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N; j <- (i+1) until N; if(kf.kernel(d.getRowTrain(i), d.getRowTrain(j)) > epsilon)){
      addBindings(map, i, j)
      if(isCountingSparsity) size2 = size2 + 2
    }
    println("The map has " + map.size + " <key,value> pairs.")
    if(isCountingSparsity) {
      println("The matrix has " + size2 + " non-sparse elements.")
    }
    map
  }

  def initializeRowColumnPairsTest(Nstart_train: Int, Nstart_test: Int, Nstop_train: Int, Nstop_test: Int): Future[MultiMap[Integer, Integer]] = {
    println("Preparing the hash map for the training set.")
    val promise = Promise[MultiMap[Integer, Integer]]
    Future{
      val map = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
      //iterate over all combinations
      for (i <- Nstart_train until Nstop_train; j <- Nstart_test until Nstop_test
           if(kf.kernel(d.getRowTrain(i), d.getRowTest(j)) > epsilon)){
        addBinding (map, i, j)
      }
      promise.success(map)
    }
    promise.future
  }

  def initializeRowColumnPairs(Nstart: Int, N: Int): Future[MultiMap[Integer, Integer]] = {
    println("Preparing the hash map for the training set.")
    val promise = Promise[MultiMap[Integer, Integer]]
    Future{
      val map = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
      //only iterate over the upper diagonal matrix
      for (i <- Nstart until N; j <- (i+1) until N; if(kf.kernel(d.getRowTrain(i), d.getRowTrain(j)) > epsilon)){
        addBindings(map, i, j)
      }
      promise.success(map)
    }
    promise.future
  }

  def initializeRowColumnPairs4Threads(): MultiMap[Integer, Integer] = {

    val N = d.getN_train
    val N1: Int = math.round(0.5 * N).toInt
    val N2: Int = calculateOptMatrixDim(N, N1)
    val N3: Int = calculateOptMatrixDim(N, N2)
    val N4: Int = N

    val N1_elements : BigInt = BigInt(N1) * BigInt(N1)
    val N2_elements : BigInt = BigInt(N2) * BigInt(N2) - N1_elements
    val N3_elements : BigInt = BigInt(N3) * BigInt(N3) - BigInt(N2) * BigInt(N2)
    val numElements = BigInt(N) * BigInt(N)
    val N4_elements : BigInt = numElements - BigInt(N3) * BigInt(N3)

    println("The relative proportion [%] of matrix elements in submatrices is:")
    println("submatrix 1: " + BigInt(100) * N1_elements / numElements +" %")
    println("submatrix 2: " + BigInt(100) * N2_elements / numElements +" %")
    println("submatrix 3: " + BigInt(100) * N3_elements / numElements +" %")
    println("submatrix 4: " + BigInt(100) * N4_elements / numElements +" %")

    val map1 = initializeRowColumnPairs(0, N1)
    val map2 = initializeRowColumnPairs(N1, N2)
    val map3 = initializeRowColumnPairs(N2, N3)
    val map4 = initializeRowColumnPairs(N3, N4)

    println("before for-comprehension")
    val map: Future[mutable.MultiMap[Integer, Integer]] = for {
      m1 <- map1
      m2 <- map2
      m3 <- map3
      m4 <- map4
    } yield (mergeMaps(Seq(m1,m2,m3,m4)))

    val result = Await.result(map, Duration(10,"minutes"))
    println("Successfully merged hash maps for the training set!")
    result
  }

  /*def mergeMaps(maps: Seq[mutable.MultiMap[Integer, Integer]]):
  MultiMap[Integer, Integer] = {
    maps.reduceLeft ((r, m) => m.foldLeft(r) {
        case (dict, (k, values)) => for(v <- values) {dict.addBinding(k,v)}
    })
  }*/

  private def calculateOptMatrixDim (N: Int, N1: Int): Int = {
    val t = BigInt(4) * BigInt(N1) * BigInt(N1) + BigInt(N) * BigInt(N)
    val diff = math.round(
      0.5 * (math.sqrt(t.doubleValue()) - 2 * N1)
    ).toInt
    assert(diff>0)
    N1 + diff
  }

  def mergeMaps (maps: Seq[mutable.MultiMap[Integer, Integer]]):
  MultiMap[Integer, Integer] = {
    maps.reduceLeft ((r, m) => mergeMMaps(r,m))
  }

  /**
    * Merges two multimaps and returns new merged map
    * @return
    */
  def mergeMMaps(mm1: mutable.MultiMap[Integer, Integer], mm2: mutable.MultiMap[Integer, Integer]):mutable.MultiMap[Integer, Integer]={
    for ( (k, vs) <- mm2; v <- vs ) mm1.addBinding(k, v)
    mm1
  }

  /**
    * The matrices K and S are
    * @param map
    * @param i
    * @param j
    * @return
    */
  private def addBindings (map: mutable.MultiMap[Integer, Integer], i: Int, j: Int) = {
    val i_ = Integer.valueOf(i)
    val j_ = Integer.valueOf(j)
    map.addBinding(i_, j_)
    map.addBinding(j_, i_)
  }

  /**
    * The matrices K and S are
    * @param map
    * @param i
    * @param j
    * @return
    */
  private def addBinding (map: mutable.MultiMap[Integer, Integer], i: Int, j: Int) = {
    val i_ = Integer.valueOf(i)
    val j_ = Integer.valueOf(j)
    map.addBinding(i_, j_)
  }

  /**
    * Prepares hash map for the training set.
    * The hash map stores all elements Training Instance => Set(Test Instance) as Integer=>Set(Integer),
    * where there is a strog enough (> epsilon) similarity between the training instance and a test instance.
    *
    * @param isCountingSparsity
    * @return
    */
  def initializeRowColumnPairsTest(isCountingSparsity: Boolean): MultiMap[Integer, Integer] = {
    println("Preparing the hash map for the test set.")
    val map: MultiMap[Integer, Integer] = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
    var size2 : Int = 0
    val N_train = d.getN_train
    val N_test = d.getN_test
    //iterate over all combinations
    for (i <- 0 until N_train; j <- 0 until N_test
         if(kf.kernel(d.getRowTrain(i), d.getRowTest(j)) > epsilon)){
      addBinding (map, i, j)
      if(isCountingSparsity) size2 = size2 + 1
    }
    println("The hash map has " + map.size + " <key,value> pairs.")
    if(isCountingSparsity) {
      println("The matrix has " + size2 + " non-sparse elements.")
    }
    map
  }

  def initializeRowColumnPairsTest4Threads(): MultiMap[Integer, Integer] = {

    println("Preparing the hash map for the test set.")
    val N_train = d.getN_train
    val N_test = d.getN_test

    val N1_train: Int = math.round(0.5 * N_train).toInt
    val N2_train: Int = calculateOptMatrixDim(N_train, N1_train)
    val N3_train: Int = calculateOptMatrixDim(N_train, N2_train)
    val N4_train: Int = N_train

    val N1_test: Int = math.round(0.5 * N_test).toInt
    val N2_test: Int = calculateOptMatrixDim(N_test, N1_test)
    val N3_test: Int = calculateOptMatrixDim(N_test, N2_test)
    val N4_test: Int = N_test

    val N1_elements : BigInt = BigInt(N1_train) * BigInt(N1_test)
    val N2_elements : BigInt = BigInt(N2_train) * BigInt(N2_test) - N1_elements
    val N3_elements : BigInt = BigInt(N3_train) * BigInt(N3_test) - BigInt(N2_test) * BigInt(N2_train)
    val numElements = BigInt(N_train) * BigInt(N_test)
    val N4_elements : BigInt = numElements - BigInt(N3_train) * BigInt(N3_test)

    println("The relative proportion [%] of matrix elements in submatrices is:")
    println("submatrix 1: " + BigInt(100) * N1_elements / numElements +" %")
    println("submatrix 2: " + BigInt(100) * N2_elements / numElements +" %")
    println("submatrix 3: " + BigInt(100) * N3_elements / numElements +" %")
    println("submatrix 4: " + BigInt(100) * N4_elements / numElements +" %")

    val map1 = initializeRowColumnPairsTest(0, 0, N1_train, N1_test)
    val map2 = initializeRowColumnPairsTest(N1_train, N1_test, N2_train, N2_test)
    val map3 = initializeRowColumnPairsTest(N2_train, N2_test, N3_train, N3_test)
    val map4 = initializeRowColumnPairsTest(N3_train, N3_test, N4_train, N4_test)

    println("before for-comprehension")
    val map: Future[mutable.MultiMap[Integer, Integer]] = for {
      m1 <- map1
      m2 <- map2
      m3 <- map3
      m4 <- map4
    } yield (mergeMaps(Seq(m1,m2,m3,m4)))

    val result = Await.result(map, Duration(10,"minutes"))
    println("Successfully merged hash maps for the test set!")
    result
  }

  /**
    * Calculates the gradient vector without storing the kernel matrix Q
    *
    * In matrix notation: Q * lambda - 1
    * For an indivual entry i of the gradient vector, this is equivalent to:
    * sum over j from 1 to N of lamba(j) * d(i) * k(i,j,) * d(j)
    * In order to avoid the constraints associated to the bias term,
    * I use a trick advocated in Christiani & Shawe-Taylor ""An Introduction to Support Vector Machines and other kernel-based learning methods"
    * 2000, pages 129-135: I add one dimension to the feature space that accounts for the bias.
    * In input space I replace x = (x1,x2,...,xn) by x_ = (x1,x2,...,xn,tau) and w = (w1,w2,...,wn) by w_ = (w1,w2,...,wn,b/tau).
    * The SVM model <w,x>+b is then replaced by <w_,x_> = x1*w1 + x2*w2 +...+ xn*wn + tau * (b/tau).
    * In the dual formulation, I replace K(x,y) by K(x,y) + tau * tau .
    * This is not without drawbacks, because the "geometric margin of the separating hyperplane in the augmented space will typically be less
    * than that in the original space." (Christiani & Shawe-Taylor, page 131)
    * I thus have to find a suitable value of tau, which is given by the maximum of the squared euclidean norm of all inputs.
    * With this choice, it is guaranteed that the fat-shattering dimension will not increase by more than factor 4 compared to input space.
    *
    *  @param alphas The current dual variables.
    *  @return
    */
  override def calculateGradient(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N = d.getN_train
    val labels: DenseVector[Double] = d.getLabelsTrain.map(x=>x.toDouble)
    val z: DenseVector[Double] = alphas *:* labels
    //for the diagonal:
    val v : DenseVector[Double] = z  *:* diagonal *:* labels
    //for the off-diagonal entries:
    for (i <- 0 until N; labelTrain = d.getLabelTrain(i); rowTrain_i = d.getRowTrain(i); setOfCols <- rowColumnPairs.get(i); j<- setOfCols){
      v(i) += z(j.toInt) * labelTrain * kf.kernel(rowTrain_i, d.getRowTrain(j))
    }
    v
  }

  override def predictOnTrainingSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N = d.getN_train
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    //for the diagonal:
    val v : DenseVector[Double] = z *:* diagonal
    //for the off-diagonal entries:
    for (i <- 0 until N; rowTrain_i = d.getRowTrain(i); setOfCols <- rowColumnPairs.get(i); j<- setOfCols){
      v(i) += z(j.toInt) * kf.kernel(rowTrain_i, d.getRowTrain(j))
    }
    signum(v)
  }

  override def predictOnTestSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N_test = d.getN_test
    val v = DenseVector.fill(N_test){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    //logClassDistribution(z)
    for ((i,set) <- rowColumnPairsTest; j <- set){
      v(j.toInt) = v(j.toInt) + z(i.toInt) * kf.kernel(d.getRowTest(j), d.getRowTrain(i))
    }
    //logClassDistribution(v)
    signum(v)
  }

  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile):
    */
  def predictOnTestSet(alphas : Alphas) : (Int,Int)  = {
    val N_test = d.getN_test
    val maxQuantile = 40
    val V = DenseMatrix.zeros[Double](maxQuantile,N_test)
    val Z = DenseMatrix.zeros[Double](maxQuantile,N_test)
    val labels = d.getLabelsTrain.map(x=>x.toDouble)

    println("Calculate Z matrix")
    val qu = DenseVector.ones[Double](maxQuantile)
    for(q <- 0 until maxQuantile; quantile : Double = 0.01 * q) {
      qu(q)= alphas.getQuantile(quantile)
    }

    //println("The quantiles: "+qu)
    //val ZZ = clip(Z(*,::), qu, 10000.0)
    //Z(*, ::) := clip(Z(*,::), 0.0, 1.0)
    //alphas.alpha.clip(a, lower, upper)

    def clip(vector : DenseVector[Double], threshold: Double) : DenseVector[Double] = {
      vector.map(x => if (x < threshold) 0 else x)
    }

    for(q <- 0 until maxQuantile; quantile : Double = 0.01 * q) {
      Z(q, ::) := (clip(alphas.alpha, qu(q))  *:* labels).t
    }

    //Z(0 until maxQuantile, ::) := Z(0 until maxQuantile, ::).t *:* labels

    println("Calculate predictions")
    for ((i,set) <- rowColumnPairsTest; j <- set; valueKernelFunction = kf.kernel(d.getRowTest(j), d.getRowTrain(i))){
      V(0 until maxQuantile,j.toInt) := V(0 until maxQuantile,j.toInt) + Z(0 until maxQuantile,i.toInt) * valueKernelFunction
    }

    println("Determine the optimal sparsity")
    //determine the optimal sparsity
    var bestCase = 0
    var bestQuantile = 0
    for(q <- 0 until maxQuantile) {
      val correctPredictions = calcCorrectPredictions(V(q, ::).t, d.getLabelsTest)
      if(correctPredictions >= bestCase){
        bestCase = correctPredictions
        bestQuantile = q
      }
    }
    assert((bestQuantile<=99) && (bestQuantile>=0),"Invalid quantile: "+bestQuantile)
    (bestQuantile, bestCase)
  }

  def calcCorrectPredictions(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    (signum(v0) *:* labels.map(x => x.toDouble) ).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  private def logClassDistribution (z: DenseVector[Double]) = {
    val positiveEntries = z.map(x => if (x > 0) 1 else 0).reduce(_ + _)
    val negativeEntries = z.map(x => if (x < 0) 1 else 0).reduce(_ + _)
    println("In vector with length: "+z.length+" there are: "+positiveEntries+" positive and: "+negativeEntries+" negative entries!")
  }
}



