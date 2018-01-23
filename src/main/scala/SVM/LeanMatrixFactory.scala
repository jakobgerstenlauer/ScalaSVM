package SVM

import breeze.linalg.{DenseVector, _}
import breeze.numerics.signum

import scala.collection.mutable
import scala.collection.mutable.{HashMap, MultiMap, Set => MSet}
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.{Failure, Success}
import SVM.DataSetType.{Test, Train, Validation}

case class LeanMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){
  /**
    * Cached diagonal of K, i.e. the kernel matrix of the training set.
    */
  val diagonal : DenseVector[Double]= initializeDiagonal()

  def initializeDiagonal():DenseVector[Double]={
    val N = d.getN_Train
    val diagonal = DenseVector.zeros[Double](N)
    for (i <- 0 until N){
      diagonal(i) = kf.kernel(d.getRow(Train,i), d.getRow(Train,i))
    }
    diagonal
  }

  /**
    * key: row index of matrix K
    * value: set of non-sparse column indices of matrix K
    */
  val rowColumnPairs : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairs4Threads()

  /**
    * key: row index of matrix S (index of validation instance)
    * value: set of non-sparse column indices of matrix S (index of trainings instance)
    */
  val rowColumnPairsValidation1 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsValidation4Threads(0)
  val rowColumnPairsValidation2 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsValidation4Threads(1)
  val rowColumnPairsValidation3 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsValidation4Threads(2)
  val rowColumnPairsValidation4 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsValidation4Threads(3)


  def hashMapNirvana(map: Future[MultiMap[Integer, Integer]]):Unit={
    map onComplete {
      case Success(m) => m.empty; m.finalize()
      case Failure(t) => println("An error when creating the hash map for the training set: " + t.getMessage)
    }
    map.finalize()
  }

  def freeValidationHashMaps() : Unit = {
    hashMapNirvana(rowColumnPairsValidation1)
    hashMapNirvana(rowColumnPairsValidation2)
    hashMapNirvana(rowColumnPairsValidation3)
    hashMapNirvana(rowColumnPairsValidation4)
  }

  /**
    * key: row index of matrix S (index of validation instance)
    * value: set of non-sparse column indices of matrix S (index of trainings instance)
    */
  val rowColumnPairsTest1 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsTest4Threads(0)
  val rowColumnPairsTest2 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsTest4Threads(1)
  val rowColumnPairsTest3 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsTest4Threads(2)
  val rowColumnPairsTest4 : Future[MultiMap[Integer, Integer]] = initializeRowColumnPairsTest4Threads(3)

  /**
    *
    * @param isCountingSparsity Should the sparsity of the matrix representation be assessed? Involves some overhead.
    * @return
    */
  def initializeRowColumnPairs(isCountingSparsity: Boolean): MultiMap[Integer, Integer] = {
    val map: MultiMap[Integer, Integer] = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
    val N = d.getN_Train
    var size2 : Int = N
    val maxIterations : Int = (N * N - N) / 2
    println("Number of iterations: "+ maxIterations)
    //only iterate over the upper diagonal matrix
    for (i <- 0 until N; j <- (i+1) until N; if(kf.kernel(d.getRow(Train,i), d.getRow(Train,j)) > epsilon)){
      addBindings(map, i, j)
      if(isCountingSparsity) size2 = size2 + 2
    }
    println("The map has " + map.size + " <key,value> pairs.")
    if(isCountingSparsity) {
      println("The matrix has " + size2 + " non-sparse elements.")
    }
    map
  }

  def initializeRowColumnPairs(Nstart_train: Int, Nstart_test: Int, Nstop_train: Int, Nstop_test: Int, dataSetType: DataSetType.Value): Future[MultiMap[Integer, Integer]] = {
    val promise = Promise[MultiMap[Integer, Integer]]
    Future{
      val map = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
      //iterate over all combinations
      for (i <- Nstart_train until Nstop_train; j <- Nstart_test until Nstop_test
           if(kf.kernel(d.getRow(Train,i), d.getRow(dataSetType,j)) > epsilon)){
        addBinding (map, i, j)
      }
      promise.success(map)
    }
    promise.future
  }

  def initializeRowColumnPairs(Nstart: Int, N: Int): Future[MultiMap[Integer, Integer]] = {
    val promise = Promise[MultiMap[Integer, Integer]]
    Future{
      val map = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
      //only iterate over the upper diagonal matrix
      for (i <- Nstart until N; j <- (i+1) until N; if(kf.kernel(d.getRow(Train,i), d.getRow(Train,j)) > epsilon)){
        addBindings(map, i, j)
      }
      promise.success(map)
    }
    promise.future
  }

  def initializeRowColumnPairs4Threads(): Future[MultiMap[Integer, Integer]] = {
    val promise = Promise[MultiMap[Integer, Integer]]
    val N = d.getN_Train
    val N1: Int = math.round(0.5 * N).toInt
    val N2: Int = calculateOptMatrixDim(N, N1)
    val N3: Int = calculateOptMatrixDim(N, N2)
    val N4: Int = N
    /*
    val N1_elements : BigInt = BigInt(N1) * BigInt(N1)
    val numElements = BigInt(N) * BigInt(N)
    val N2_elements : BigInt = BigInt(N2) * BigInt(N2) - N1_elements
    val N3_elements : BigInt = BigInt(N3) * BigInt(N3) - BigInt(N2) * BigInt(N2)
    val N4_elements : BigInt = numElements - BigInt(N3) * BigInt(N3)
    println("The relative proportion [%] of matrix elements in submatrices is:")
    println("submatrix 1: " + BigInt(100) * N1_elements / numElements +" %")
    println("submatrix 2: " + BigInt(100) * N2_elements / numElements +" %")
    println("submatrix 3: " + BigInt(100) * N3_elements / numElements +" %")
    println("submatrix 4: " + BigInt(100) * N4_elements / numElements +" %")*/
    val map1 = initializeRowColumnPairs(0, N1)
    val map2 = initializeRowColumnPairs(N1, N2)
    val map3 = initializeRowColumnPairs(N2, N3)
    val map4 = initializeRowColumnPairs(N3, N4)

    val map: Future[mutable.MultiMap[Integer, Integer]] = for {
      m1 <- map1
      m2 <- map2
      m3 <- map3
      m4 <- map4
    } yield (mergeMaps(Seq(m1,m2,m3,m4)))

    map onComplete {
      case Success(mergedHashMap) => promise.success(mergedHashMap);
      case Failure(t) => println("An error when creating the hash map for the training set: " + t.getMessage)
    }
    promise.future
  }

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
    * The hash map stores all elements Training Instance => Set(Validation Instance) as Integer=>Set(Integer),
    * where there is a strog enough (> epsilon) similarity between the training instance and a validation set instance.
    *
    * @param isCountingSparsity
    * @return
    */
  def initializeRowColumnPairsValidation(isCountingSparsity: Boolean): MultiMap[Integer, Integer] = {
    val map: MultiMap[Integer, Integer] = new HashMap[Integer, MSet[Integer]] with MultiMap[Integer, Integer]
    var size2 : Int = 0
    val N_train = d.getN_Train
    val N_test = d.getN_Validation
    //iterate over all combinations
    for (i <- 0 until N_train; j <- 0 until N_test
         if(kf.kernel(d.getRow(Train,i), d.getRow(Validation,j)) > epsilon)){
      addBinding (map, i, j)
      if(isCountingSparsity) size2 = size2 + 1
    }
    println("The hash map has " + map.size + " <key,value> pairs.")
    if(isCountingSparsity) {
      println("The matrix has " + size2 + " non-sparse elements.")
    }
    map
  }

  /**
    *
    * @param replicate Replicates of the validation set, must be exactly 0,1,2,3!!!
    * @return
    */
  def initializeRowColumnPairsValidation4Threads (replicate: Int): Future[MultiMap[Integer, Integer]] = {
    assert(replicate>=0 && replicate<=3, "The 4 replicates of the validation set must be coded as 0,1,2,3!")
    val promise = Promise[MultiMap[Integer, Integer]]
    val N_train = d.getN_Train
    //The validation set is split into 4 separate (equal sized) validation sets
    assert(d.getN_Validation % 4 == 0, "The nr of observations in the validation set must be a multiple of 4!")
    val N_val : Int = d.getN_Validation / 4
    val N1_train: Int = math.round(0.5 * N_train).toInt
    val N2_train: Int = calculateOptMatrixDim(N_train, N1_train)
    val N3_train: Int = calculateOptMatrixDim(N_train, N2_train)
    val N4_train: Int = N_train
    val N1_val: Int = math.round(0.5 * N_val).toInt
    val N2_val: Int = calculateOptMatrixDim(N_val, N1_val)
    val N3_val: Int = calculateOptMatrixDim(N_val, N2_val)
    val N4_val: Int = N_val

    /*val N1_elements : BigInt = BigInt(N1_train) * BigInt(N1_val)
    val N2_elements : BigInt = BigInt(N2_train) * BigInt(N2_val) - N1_elements
    val N3_elements : BigInt = BigInt(N3_train) * BigInt(N3_val) - BigInt(N2_val) * BigInt(N2_train)
    val numElements = BigInt(N_train) * BigInt(N_val)
    val N4_elements : BigInt = numElements - BigInt(N3_train) * BigInt(N3_val)
    println("The relative proportion [%] of matrix elements in submatrices is:")
    println("submatrix 1: " + BigInt(100) * N1_elements / numElements +" %")
    println("submatrix 2: " + BigInt(100) * N2_elements / numElements +" %")
    println("submatrix 3: " + BigInt(100) * N3_elements / numElements +" %")
    println("submatrix 4: " + BigInt(100) * N4_elements / numElements +" %")*/

    //For the validation set, I have to add an offset term because I create 4 separate matrices:
    val offset = N_val * replicate
    val map1 = initializeRowColumnPairs(0,             0+offset, N1_train, N1_val+offset, Validation)
    val map2 = initializeRowColumnPairs(N1_train, N1_val+offset, N2_train, N2_val+offset, Validation)
    val map3 = initializeRowColumnPairs(N2_train, N2_val+offset, N3_train, N3_val+offset, Validation)
    val map4 = initializeRowColumnPairs(N3_train, N3_val+offset, N4_train, N4_val+offset, Validation)

    val map: Future[mutable.MultiMap[Integer, Integer]] = for {
      m1 <- map1
      m2 <- map2
      m3 <- map3
      m4 <- map4
    } yield (mergeMaps(Seq(m1,m2,m3,m4)))

    map onComplete {
      case Success(mergedHashMap) => promise.success(mergedHashMap);
      case Failure(t) => println("An error when creating the hash map for the validation set: " + t.getMessage)
    }
    promise.future
  }

  /**
    *
    * @param replicate Replicates of the validation set, must be exactly 0,1,2,3!!!
    * @return
    */
  def initializeRowColumnPairsTest4Threads (replicate: Int): Future[MultiMap[Integer, Integer]] = {
    assert(replicate>=0 && replicate<=3, "The 4 replicates of the validation set must be coded as 0,1,2,3!")
    val promise = Promise[MultiMap[Integer, Integer]]
    val N_train = d.getN_Train
    //The test set is split into 4 separate (equal sized) validation sets
    assert(d.getN_Test % 4 == 0, "The nr of observations in the test set must be a multiple of 4!")
    val N_test : Int = d.getN_Test / 4

    val N1_train: Int = math.round(0.5 * N_train).toInt
    val N2_train: Int = calculateOptMatrixDim(N_train, N1_train)
    val N3_train: Int = calculateOptMatrixDim(N_train, N2_train)
    val N4_train: Int = N_train

    val N1_test: Int = math.round(0.5 * N_test).toInt
    val N2_test: Int = calculateOptMatrixDim(N_test, N1_test)
    val N3_test: Int = calculateOptMatrixDim(N_test, N2_test)
    val N4_test: Int = N_test
    /*
    val N1_elements : BigInt = BigInt(N1_train) * BigInt(N1_test)
    val N2_elements : BigInt = BigInt(N2_train) * BigInt(N2_test) - N1_elements
    val N3_elements : BigInt = BigInt(N3_train) * BigInt(N3_test) - BigInt(N2_test) * BigInt(N2_train)
    val numElements = BigInt(N_train) * BigInt(N_test)
    val N4_elements : BigInt = numElements - BigInt(N3_train) * BigInt(N3_test)
    println("The relative proportion [%] of matrix elements in submatrices is:")
    println("submatrix 1: " + BigInt(100) * N1_elements / numElements +" %")
    println("submatrix 2: " + BigInt(100) * N2_elements / numElements +" %")
    println("submatrix 3: " + BigInt(100) * N3_elements / numElements +" %")
    println("submatrix 4: " + BigInt(100) * N4_elements / numElements +" %")*/

    //For the validation set, I have to add an offset term because I create 4 separate matrices:
    val offset = N_test * replicate

    val map1 = initializeRowColumnPairs(0,             0+offset, N1_train, N1_test+offset, Test)
    val map2 = initializeRowColumnPairs(N1_train, N1_test+offset, N2_train, N2_test+offset, Test)
    val map3 = initializeRowColumnPairs(N2_train, N2_test+offset, N3_train, N3_test+offset, Test)
    val map4 = initializeRowColumnPairs(N3_train, N3_test+offset, N4_train, N4_test+offset, Test)

    val map: Future[mutable.MultiMap[Integer, Integer]] = for {
      m1 <- map1
      m2 <- map2
      m3 <- map3
      m4 <- map4
    } yield (mergeMaps(Seq(m1,m2,m3,m4)))

    map onComplete {
      case Success(mergedHashMap) => promise.success(mergedHashMap);
      case Failure(t) => println("An error when creating the hash map for the validation set: " + t.getMessage)
    }
    promise.future
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
    val hashMap = Await.result(rowColumnPairs, Duration(60,"minutes"))
    val N = d.getN_Train
    val labels: DenseVector[Double] = d.getLabels(Train).map(x=>x.toDouble)
    val z: DenseVector[Double] = alphas *:* labels
    //for the diagonal:
    val v : DenseVector[Double] = z  *:* diagonal *:* labels
    //for the off-diagonal entries:
    for (i <- 0 until N; labelTrain = d.getLabel(Train,i); rowTrain_i = d.getRow(Train,i); setOfCols <- hashMap.get(i); j<- setOfCols){
      v(i) += z(j.toInt) * labelTrain * kf.kernel(rowTrain_i, d.getRow(Train,j))
    }
    v
  }

  override def predictOnTrainingSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val hashMap = Await.result(rowColumnPairs, Duration(60,"minutes"))
    val N = d.getN_Train
    val z : DenseVector[Double] = alphas *:* d.getLabels(Train).map(x=>x.toDouble)
    //for the diagonal:
    val v : DenseVector[Double] = z *:* diagonal
    //for the off-diagonal entries:
    for (i <- 0 until N; rowTrain_i = d.getRow(Train,i); setOfCols <- hashMap.get(i); j<- setOfCols){
      v(i) += z(j.toInt) * kf.kernel(rowTrain_i, d.getRow(Train,j))
    }
    signum(v)
  }

  private def getHashMapValidation(replicate: Int):Future[MultiMap[Integer, Integer]]={
    replicate match{
      case 0 => rowColumnPairsValidation1
      case 1 => rowColumnPairsValidation2
      case 2 => rowColumnPairsValidation3
      case 3 => rowColumnPairsValidation4
      case _ =>  throw new IllegalArgumentException("Unsupported replicate nr!")
    }
  }

  private def getHashMapTest(replicate: Int):Future[MultiMap[Integer, Integer]]={
    replicate match{
      case 0 => rowColumnPairsTest1
      case 1 => rowColumnPairsTest2
      case 2 => rowColumnPairsTest3
      case 3 => rowColumnPairsTest4
      case _ =>  throw new IllegalArgumentException("Unsupported replicate nr!")
    }
  }

  def predictOnValidationSet (alphas : DenseVector[Double], replicate: Int) : DenseVector[Double]  = {
    val hashMapPromise = getHashMapValidation(replicate)
    val hashMap = Await.result(hashMapPromise, Duration(60,"minutes"))
    val N_validation = d.getN_Validation
    val v = DenseVector.fill(N_validation){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabels(Train).map(x=>x.toDouble)
    //logClassDistribution(z)
    for ((i,set) <- hashMap; j <- set){
      v(j.toInt) = v(j.toInt) + z(i.toInt) * kf.kernel(d.getRow(Validation,j), d.getRow(Train,i))
    }
    //logClassDistribution(v)
    signum(v)
  }

  def predictOnValidationSet (alphas : Alphas, replicate: Int, maxQuantile: Int) : Future[DenseVector[Int]]  = {
    val promise = Promise[DenseVector[Int]]
    Future{
      val hashMapPromise = getHashMapValidation(replicate)
      val hashMap = Await.result(hashMapPromise, Duration(60,"minutes"))
      val N_validation = d.getN_Validation
      val N_train = d.getN_Train
      val V = DenseMatrix.zeros[Double](maxQuantile+1,N_validation)
      val Z = DenseMatrix.zeros[Double](maxQuantile+1,N_train)
      val labels = d.getLabels(Train).map(x=>x.toDouble)
      val qu = DenseVector.ones[Double](maxQuantile)
      for(q <- 0 until maxQuantile; quantile : Double = 0.01 * q) {
        qu(q)= alphas.getQuantile(quantile)
      }
      def clip(vector : DenseVector[Double], threshold: Double) : DenseVector[Double] = {
        vector.map(x => if (x < threshold) 0 else x)
      }
      for(q <- 0 until maxQuantile) {
        Z(q, ::) := (clip(alphas.alpha, qu(q))  *:* labels).t
      }
      for ((i,set) <- hashMap; j <- set; valueKernelFunction = kf.kernel(d.getRow(Validation,j), d.getRow(Train,i))){
        V(0 to maxQuantile,j.toInt) := V(0 to maxQuantile,j.toInt) + Z(0 to maxQuantile,i.toInt) * valueKernelFunction
      }
      val correctPredictions = DenseVector.zeros[Int](maxQuantile+1)
      for(q <- 0 to maxQuantile) {
        correctPredictions(q) = calcCorrectPredictions(V(q, ::).t, d.getLabels(Validation))
      }
      promise.success(correctPredictions)
    }
    promise.future
  }

  private def predictOnTestSet (alphas : Alphas, replicate: Int) : Future[Int]  = {
    val promise = Promise[Int]
    Future{
      val hashMapPromise = getHashMapTest(replicate)
      val hashMap = Await.result(hashMapPromise, Duration(60,"minutes"))
      val N_test = d.getN_Test
      val N_train = d.getN_Train
      val v = DenseVector.fill(N_test){0.0}
      val z : DenseVector[Double] = alphas.alpha *:* d.getLabels(Train).map(x=>x.toDouble)
      for ((i,set) <- hashMap; j <- set; valueKernelFunction = kf.kernel(d.getRow(Test,j), d.getRow(Train,i))){
        v(j.toInt) = v(j.toInt) + z(i.toInt) * valueKernelFunction
      }
      val correctPredictions = calcCorrectPredictions(signum(v), d.getLabels(Test))
      promise.success(correctPredictions)
    }
    promise.future
  }
  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile):
    */
  def predictOnValidationSet (alphas : Alphas, iteration: Int) : Future[(Int,Int,Int)] = {
    val promise = Promise[(Int,Int,Int)]
    val maxQuantile = 99

    val predict1 = predictOnValidationSet(alphas.copy(), 0, maxQuantile)
    val predict2 = predictOnValidationSet(alphas.copy(), 1, maxQuantile)
    val predict3 = predictOnValidationSet(alphas.copy(), 2, maxQuantile)
    val predict4 = predictOnValidationSet(alphas.copy(), 3, maxQuantile)

    //Merge the results of the four threads by simply summing the vectors
    val futureSumCorrectPredictions: Future[DenseVector[Int]] = for {
      p1 <- predict1
      p2 <- predict2
      p3 <- predict3
      p4 <- predict4
    } yield (p1 + p2 + p3 + p4)

    futureSumCorrectPredictions onComplete {
      case Success(correctPredictions) => {
        //println("Determine the optimal sparsity")
        //determine the optimal sparsity measured as the sum of correct predictions over all 4 validation subsets
        var bestCase = 0
        var bestQuantile = 0
        for(q <- 0 until maxQuantile) {
          //println("Debug: Quantile "+ q +" correct predictions: " + correctPredictions(q))
          val sumCorrectPredictions = correctPredictions(q)
          if(sumCorrectPredictions >= bestCase){
            bestCase = sumCorrectPredictions
            bestQuantile = q
            //println("Debug: bestCase "+ bestCase +" bestQuantile: " + bestQuantile)
          }
        }
        assert((bestQuantile<=99) && (bestQuantile>=0),"Invalid quantile: "+bestQuantile)
        println("Accuracy validation set: "+bestCase +"/"+ getData().getN_Validation+" with sparsity: "+ bestQuantile)
        promise.success((bestQuantile,bestCase, iteration))
      }
      case Failure(t) => {
        println("An error occurred when trying to calculate the correct predictions for all quantiles and validation subsets!"
          + t.getMessage)
        promise.failure(t.getCause)
      }
    }
    promise.future
  }

  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile):
    */
  def predictOnTestSet (alphas : Alphas) : Future[Int] = {
    val promise = Promise[Int]
    val predict1 = predictOnTestSet(alphas.copy(), 0)
    val predict2 = predictOnTestSet(alphas.copy(), 1)
    val predict3 = predictOnTestSet(alphas.copy(), 2)
    val predict4 = predictOnTestSet(alphas.copy(), 3)

    //Merge the results of the four threads by simply summing the vectors
    val futureSumCorrectPredictions: Future[Int] = for {
      p1 <- predict1
      p2 <- predict2
      p3 <- predict3
      p4 <- predict4
    } yield (p1 + p2 + p3 + p4)

    futureSumCorrectPredictions onComplete {
      case Success(correctPredictions) => {
        println("Nr of correct predictions for test set: "+correctPredictions +"/"+ getData().getN_Test)
        promise.success(correctPredictions)
      }
      case Failure(t) => {
        println("An error occurred when trying to calculate the correct predictions for the test set!"
          + t.getMessage)
        promise.failure(t.getCause)
      }
    }
    //Await.result(futureSumCorrectPredictions, Duration(60,"minutes"))
    promise.future
  }

  def calcCorrectPredictions(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    (signum(v0) *:* labels.map(x => x.toDouble) ).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  private def logClassDistribution (z: DenseVector[Double]) = {
    val positiveEntries = z.map(x => if (x > 0) 1 else 0).reduce(_ + _)
    val negativeEntries = z.map(x => if (x < 0) 1 else 0).reduce(_ + _)
    println("In vector with length: "+z.length+" there are: "+positiveEntries+" positive and: "+negativeEntries+" negative entries!")
  }
}