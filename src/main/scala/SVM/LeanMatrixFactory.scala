package SVM

import breeze.linalg.{DenseVector, _}
import breeze.numerics.{exp, signum}

import scala.collection.mutable.{HashMap, MultiMap, Set => MSet}
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.{Failure, Success}
import SVM.DataSetType.{Test, Train, Validation}
import breeze.plot.{Figure,plot}

import scala.collection.mutable.ListBuffer

object LeanMatrixFactory{
  val maxDuration = Duration(12*60,"minutes")
}

case class LeanMatrixFactory(d: Data, kf: KernelFunction, epsilon: Double) extends BaseMatrixFactory(d, kf, epsilon){
  /**
    * Cached diagonal of K, i.e. the kernel matrix of the training set.
    */
  val diagonal : DenseVector[Double]= initializeDiagonal()

  def initializeDiagonal():DenseVector[Double]={
    val N = d.getN_Train
    val diagonal = DenseVector.zeros[Double](N)
    for (i <- 0 until N; x = d.getRow(Train,i)){
      diagonal(i) = kf.kernel(x, x)
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
      case Success(m) => m.empty
      case Failure(t) => println("An error when creating the hash map for the training set: " + t.getMessage)
    }
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


  /**
    *
    * This function is intended for the non-symmetric matrices of the validation and the test set.
    * @param Nstart_train
    * @param Nstart_test
    * @param Nstop_train
    * @param Nstop_test
    * @param dataSetType
    * @return
    */
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

  /**
    * This function is intended for the symmetric training set matrix.
    * @param Nstart
    * @param N
    * @return
    */
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

    val map: Future[MultiMap[Integer, Integer]] = for {
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

  def mergeMaps (maps: Seq[MultiMap[Integer, Integer]]):
  MultiMap[Integer, Integer] = {
    maps.reduceLeft ((r, m) => mergeMMaps(r,m))
  }

  /**
    * Merges two multimaps and returns new merged map
    * @return
    */
  def mergeMMaps(mm1: MultiMap[Integer, Integer], mm2: MultiMap[Integer, Integer]):MultiMap[Integer, Integer]={
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
  private def addBindings (map: MultiMap[Integer, Integer], i: Int, j: Int) = {
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
  private def addBinding (map: MultiMap[Integer, Integer], i: Int, j: Int) = {
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

    val map: Future[MultiMap[Integer, Integer]] = for {
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

    val map: Future[MultiMap[Integer, Integer]] = for {
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
    val promise = Promise[DenseVector[Double]]
    Future {
      val hashMap = Await.result(rowColumnPairs, LeanMatrixFactory.maxDuration)
      val N = d.getN_Train
      val labels: DenseVector[Double] = d.getLabels(Train).map(x => x.toDouble)
      val z: DenseVector[Double] = alphas *:* labels

      val N1:Int = N/4
      val N2:Int = N/2
      val N3:Int = 3*(N/4)

      val gradient1 = calculateGradient(0, N1, N, hashMap, z)
      val gradient2 = calculateGradient(N1, N2, N, hashMap, z)
      val gradient3 = calculateGradient(N2, N3, N, hashMap, z)
      val gradient4 = calculateGradient(N3, N, N, hashMap, z)

      //Merge the results of the four threads by simply summing the vectors
      val futureSumGradients: Future[DenseVector[Double]] = for {
        p1 <- gradient1
        p2 <- gradient2
        p3 <- gradient3
        p4 <- gradient4
      } yield (p1 + p2 + p3 + p4)

      //for the diagonal:
      val v: DenseVector[Double] = z *:* diagonal *:* labels

      futureSumGradients onComplete {
        case Success(sumGradients) => {
          promise.success(sumGradients + v)
        }
        case Failure(t) => {
          println("An error occurred when trying to calculate the sum of gradients!"
            + t.getMessage)
          promise.failure(t.getCause)
        }
      }
    }
    Await.result(promise.future, LeanMatrixFactory.maxDuration)
  }


  def calculateGradient(N_start : Int, N_stop : Int, N: Int, hashMap : MultiMap[Integer, Integer], z: DenseVector[Double]): Future[DenseVector[Double]]  = {
    val promise = Promise[DenseVector[Double]]
    Future{
      val v = DenseVector.zeros[Double](N)
      //for the off-diagonal entries:
      for (i <- N_start until N_stop; labelTrain = d.getLabel(Train,i); rowTrain_i = d.getRow(Train,i); setOfCols <- hashMap.get(i); j<- setOfCols){
        v(i) += z(j.toInt) * labelTrain * kf.kernel(rowTrain_i, d.getRow(Train,j))
      }
      promise.success(v)
    }
    promise.future
  }

  private def getHashMap(dataSetType: DataSetType.Value, replicate: Int):Future[MultiMap[Integer, Integer]]={

    def getHashMapValidation(replicate: Int):Future[MultiMap[Integer, Integer]]={
      replicate match{
        case 0 => rowColumnPairsValidation1
        case 1 => rowColumnPairsValidation2
        case 2 => rowColumnPairsValidation3
        case 3 => rowColumnPairsValidation4
        case _ =>  throw new IllegalArgumentException("Unsupported replicate nr!")
      }
    }

    def getHashMapTest(replicate: Int):Future[MultiMap[Integer, Integer]]={
      replicate match{
        case 0 => rowColumnPairsTest1
        case 1 => rowColumnPairsTest2
        case 2 => rowColumnPairsTest3
        case 3 => rowColumnPairsTest4
        case _ =>  throw new IllegalArgumentException("Unsupported replicate nr!")
      }
    }

    dataSetType match{
      case Test => getHashMapTest(replicate)
      case Validation => getHashMapValidation(replicate)
      case _ =>  throw new IllegalArgumentException("Unsupported data set type!")
    }
  }

  def predictOnValidationSet (alphas : Alphas, replicate: Int, maxQuantile: Int) : Future[DenseVector[Int]]  = {
    val promise = Promise[DenseVector[Int]]
    Future{
      val hashMapPromise = getHashMap(Validation,replicate)
      val hashMap = Await.result(hashMapPromise, LeanMatrixFactory.maxDuration)
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

  /**
    * Calculates ranks for a vector.
    * If there are ties, the respective elements are assigned the same rank.
    * An example:
    * val x2 = DenseVector(1.0,1.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0)
    * findRank(x2)
    * > DenseVector(9.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
    * @param x
    * @return
    */
  def getRank(x: DenseVector[Double]) : DenseVector[Double] = {
    //val r = DenseVector.zeros(score.length)
    val ranks = DenseVector((0 until x.length).toList.sortWith( (left, right) => x(left) > x(right)).toArray)
    var j=0
    val r=DenseVector.zeros[Double](x.length)
    for(i <- 0 until x.length){
      if (x(ranks(i)) != x(ranks(j)) ){
        j=i
      }
      r(ranks(i))=j
    }
    r
  }

  /**
    * Calculates first the ranks for a vector and then divides all ranks by the maximum rank.
    * If there are ties, the respective elements are assigned the same rank.
    * @param x
    * @return
    */
  def getQuantiles(x: DenseVector[Double]) : DenseVector[Double] = {
    //val r = DenseVector.zeros(score.length)
    val ranks = DenseVector((0 until x.length).toList.sortWith( (left, right) => x(left) > x(right)).toArray)
    var j=0
    var rankMax : Double = 0.0
    val r=DenseVector.zeros[Double](x.length)
    for(i <- 0 until x.length){
      if (x(ranks(i)) != x(ranks(j)) ){
        j=i
        rankMax=i
      }
      r(ranks(i))=j
    }
    assert(rankMax>0)
    r / rankMax
  }

  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile, iteration):
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

  def predictOnValidationSet (alphas : DenseVector[Double], replicate: Int) : DenseVector[Double]  = {
    val hashMapPromise = getHashMap(Validation,replicate)
    val hashMap = Await.result(hashMapPromise, LeanMatrixFactory.maxDuration)
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

  override def predictOnTrainingSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val hashMap = Await.result(rowColumnPairs, LeanMatrixFactory.maxDuration)
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

  private def predictOn(dataType: SVM.DataSetType.Value, alphas : Alphas,
                        hashMap: MultiMap[Integer, Integer]) : Future[DenseVector[Double]]  = {
    val promise = Promise[DenseVector[Double]]
    Future{
      println("predictOn "+dataType.toString+ " and hash map of size: "+hashMap.size)
      val N = d.getN(dataType)
      println("N: "+N)
      val v = DenseVector.fill(N){0.0}
      val z : DenseVector[Double] = alphas.alpha *:* d.getLabels(Train).map(x=>x.toDouble)
      for ((i,set) <- hashMap; j <- set; valueKernelFunction = kf.kernel(d.getRow(dataType,j), d.getRow(Train,i))){
        v(j.toInt) = v(j.toInt) + z(i.toInt) * valueKernelFunction
      }
      println("promise fulfilled!")
      promise.success(v)
    }
    promise.future
  }

  private def predictOn(dataType: SVM.DataSetType.Value, alphas : Alphas, hashMap: MultiMap[Integer,Integer], threshold: Double) : Future[DenseVector[Double]]  = {
    assert(threshold>0.0 && threshold<1.0)
    val promise = Promise[DenseVector[Double]]
    Future{
      val N = d.getN(dataType)
      val v = DenseVector.fill(N){0.0}
      val z : DenseVector[Double] = alphas.alpha *:* d.getLabels(Train).map(x=>x.toDouble)
      for ((i,set) <- hashMap; j <- set; valueKernelFunction = kf.kernel(d.getRow(dataType,j), d.getRow(Train,i))){
        v(j.toInt) = v(j.toInt) + z(i.toInt) * valueKernelFunction
      }
      promise.success(v)
    }
    promise.future
  }

  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile):
    */
  def predictOn(dataType: SVM.DataSetType.Value, alphas : Alphas, threshold: Double) : Future[Int] = {
    assert(threshold > 0 && threshold < 1.0)
    val promise = Promise[Int]
    //Here the futures for the four hash maps for the test set replicates
    //are combined into a single future.
    val combinedFuture: Future[(MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer])] = extractHashMapPromises(dataType)

    //Once all hash maps have been created.
    combinedFuture onComplete{
      case Success(maps) =>{

        val predict1 = predictOn(dataType, alphas.copy(), maps._1, threshold)
        val predict2 = predictOn(dataType, alphas.copy(), maps._2, threshold)
        val predict3 = predictOn(dataType, alphas.copy(), maps._3, threshold)
        val predict4 = predictOn(dataType, alphas.copy(), maps._4, threshold)

        //Merge the results of the four threads by simply summing the vectors
        val v: Future[DenseVector[Double]] = for {
          p1 <- predict1
          p2 <- predict2
          p3 <- predict3
          p4 <- predict4
        } yield (p1 + p2 + p3 + p4)

        v onComplete {//Once all predictions have been calculated.
          case Success(v) => {
            val predictions = getQuantiles(v).map(x => if (x > threshold) -1.0 else +1.0)
            val correctPredictions = calcCorrectPredictions(predictions, d.getLabels(dataType))
            val correctPredictionsClass1 = calcCorrectPredictionsClass1(signum(v), d.getLabels(dataType))
            val correctPredictionsClass2 = calcCorrectPredictionsClass2(signum(v), d.getLabels(dataType))
            println("Nr of correct predictions for test set: " + correctPredictions + "/" + getData().getN(dataType))
            println("Nr of correct predictions for class 1 (+1) in test set: " + correctPredictionsClass1 + "/" + getData().getN(dataType))
            println("Nr of correct predictions for class 2 (-1) test set: " + correctPredictionsClass2 + "/" + getData().getN(dataType))
            promise.success(correctPredictions)
          }
          case Failure(t) => {
            println("An error occurred when trying to calculate the correct predictions for the test set!"
              + t.getMessage)
            promise.failure(t.getCause)
          }
        }
      }
      case Failure(t) => {
        println("An error occurred when trying to create the hash maps for the four test set replicates!"
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
  def predictOn(dataType: SVM.DataSetType.Value, alphas : Alphas) : Future[Int] = {
    val promise = Promise[Int]
    //Here the futures for the four hash maps for the test set replicates
    //are combined into a single future.
    val combinedFuture: Future[(MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer])] = extractHashMapPromises(dataType)

    //Once all hash maps have been created.
    combinedFuture onComplete{
      case Success(maps) =>{

        println("All hash maps are there!")
        val predict1 = predictOn(dataType, alphas.copy(), maps._1)
        val predict2 = predictOn(dataType, alphas.copy(), maps._2)
        val predict3 = predictOn(dataType, alphas.copy(), maps._3)
        val predict4 = predictOn(dataType, alphas.copy(), maps._4)

        //Merge the results of the four threads by simply summing the vectors
        val v: Future[DenseVector[Double]] = for {
          p1 <- predict1
          p2 <- predict2
          p3 <- predict3
          p4 <- predict4
        } yield (p1 + p2 + p3 + p4)

        v onComplete {//Once all predictions have been calculated.
          case Success(v) => {
            val correctPredictions = calcCorrectPredictions(signum(v), d.getLabels(dataType))
            val correctPredictionsClass1 = calcCorrectPredictionsClass1(signum(v), d.getLabels(dataType))
            val correctPredictionsClass2 = calcCorrectPredictionsClass2(signum(v), d.getLabels(dataType))
            println("Nr of correct predictions for " + dataType.toString() + " set: "+correctPredictions +"/"+ getData().getN(dataType))
            println("Nr of correct predictions for class 1 (+1) in " + dataType.toString() + " set: "+correctPredictionsClass1 +"/"+ getData().getN(dataType))
            println("Nr of correct predictions for class 2 (-1) in " + dataType.toString() + " set: "+correctPredictionsClass2 +"/"+ getData().getN(dataType))
            promise.success(correctPredictions)
          }
          case Failure(t) => {
            println("An error occurred when trying to calculate the correct predictions for the test set!"
              + t.getMessage)
            promise.failure(t.getCause)
          }
        }
      }
      case Failure(t) => {
        println("An error occurred when trying to create the hash maps for the four test set replicates!"
          + t.getMessage)
        promise.failure(t.getCause)
      }
    }
    promise.future
  }

  //https://stackoverflow.com/questions/11106886/scala-doubles-and-precision
  def roundAt(p: Int)(n: Double): Double = { val s = math pow (10, p); (math round n * s) / s }

  /**
    *
    * @param alphas
    * @return Tuple (optimal sparsity, nr of correct predictions for this quantile):
    */
  def predictOnAUC (dataType: SVM.DataSetType.Value, alphas : Alphas) : Future[Int] = {
    val promise = Promise[Int]

    val combinedFuture: Future[(MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer], MultiMap[Integer, Integer])] = extractHashMapPromises(dataType)

    combinedFuture onComplete{
      case Success(maps) =>{
        println("All hash maps are there!")
        val predict1 = predictOn(dataType, alphas.copy(), maps._1)
        val predict2 = predictOn(dataType, alphas.copy(), maps._2)
        val predict3 = predictOn(dataType, alphas.copy(), maps._3)
        val predict4 = predictOn(dataType, alphas.copy(), maps._4)

        //Merge the results of the four threads by simply summing the vectors
        val v: Future[DenseVector[Double]] = for {
          p1 <- predict1
          p2 <- predict2
          p3 <- predict3
          p4 <- predict4
        } yield (p1 + p2 + p3 + p4)

        v onComplete {
          case Success(v) => {
            val quantiles = getQuantiles(v)
            val correctPredictions = DenseVector.zeros[Int](99)
            val truePositives = DenseVector.zeros[Int](99)
            val falsePositives = DenseVector.zeros[Int](99)
            val truePositiveRate = DenseVector.zeros[Double](99)
            val falsePositiveRate = DenseVector.zeros[Double](99)
            val labels = d.getLabels(dataType).map(x=>if(x>0) 1 else -1)

            val numPositive = d.getLabels(dataType).map(x=>if(x>0)1 else 0).reduce(_+_)
            val numNegatives = d.getLabels(dataType).map(x=>if(x<0)1 else 0).reduce(_+_)
            val total = numPositive + numNegatives
            println("+:"+numPositive+" -:"+numNegatives)
            println("True (+) and false(-) positive rate, Q=+/sqrt(-), and total accuracy (A) for the test set and all percentiles: ")
            println("%\t+\t\t-\tQ\tA")

            var i = 0
            for(threshold <- 1 to 99){
              val cutOff : Double = threshold/100.0
              val predictions = quantiles.map(x=>if(x<cutOff) +1.0 else -1.0)
              val NumPredictionsPositive = predictions.map(x=>if(x>0) 1 else 0).reduce(_+_)

              val NumTruePositives  = calcTruePositives(predictions, labels)
              truePositives(i) = NumTruePositives
              val truePosRate = NumTruePositives / numPositive.toDouble
              truePositiveRate(i) = truePosRate

              val NumFalsePositives = calcFalsePositives(predictions, labels)
              falsePositives(i) = NumFalsePositives
              //probability of false alarm
              val falsePosRate = NumFalsePositives / NumPredictionsPositive.toDouble
              falsePositiveRate(i) = falsePosRate

              val q = if (truePosRate>0 && falsePosRate>0) truePosRate / Math.sqrt(falsePosRate) else 0.0
              val correctPredictions = calcCorrectPredictions(predictions, labels)
              val accuracy = correctPredictions / total.toDouble

              if(falsePosRate==0.0){
                println(threshold+"\t"
                  +roundAt(3)(truePosRate)+"\t\t0.0\tNAN\t"
                  +roundAt(3)(accuracy))
              }else {
                println(threshold + "\t"
                  + roundAt(3)(truePosRate) + "\t"
                  + roundAt(3)(falsePosRate) + "\t"
                  + roundAt(3)(q) + "\t"
                  + roundAt(3)(accuracy))
              }
              i = i+1
            }

            val fig = Figure()
            val plt = fig.subplot(0)
            plt += plot(falsePositiveRate, truePositiveRate, '+', name = "Receiver Operating Characteristic (ROC) curve")
            val comparisonLine = (1 to 99).map(x=> x.toDouble / 100.0)
            plt += plot(comparisonLine, comparisonLine, '-', name = "Benchmark random classifier")
            //plt.plot.addAnnotation(new XYTextAnnotation(txt, 3890.0, 200.0))
            plt.xlabel = "False negative rate."
            plt.ylabel = "True positive rate."
            plt.legend = true
            fig.refresh()
            fig.saveas("ROC_curve.png",400)

            promise.success(correctPredictions.reduce((a,b)=>max(a,b)))
          }
          case Failure(t) => {
            println("An error occurred when trying to calculate the correct predictions for the test set!"
              + t.getMessage)
            promise.failure(t.getCause)
          }
        }
      }
      case Failure(t) => {
        println("An error occurred when trying to create the hash maps for the four test set replicates!"
          + t.getMessage)
        promise.failure(t.getCause)
      }
    }
    promise.future
  }

  private def extractHashMapPromises(dataType: SVM.DataSetType.Value) = {
    val hashMapPromise0 = getHashMap(dataType, 0)
    val hashMapPromise1 = getHashMap(dataType, 1)
    val hashMapPromise2 = getHashMap(dataType, 2)
    val hashMapPromise3 = getHashMap(dataType, 3)

    val combinedFuture = for {
      map0 <- hashMapPromise0
      map1 <- hashMapPromise1
      map2 <- hashMapPromise2
      map3 <- hashMapPromise3
    } yield (map0, map1, map2, map3)
    combinedFuture
  }

  def calcCorrectPredictions (v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    (signum(v0) *:* labels.map(x => x.toDouble) ).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  def calcTruePositives(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    val x : BitVector = labels >:> 0
    val y : BitVector = v0 >:> 0.0
    val z = x & y
    z.map(x => if (x) 1 else 0).reduce(_+_)
  }

  def calcFalsePositives(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    val x : BitVector = labels<:< 0
    val y : BitVector = v0>:> 0.0
    val z = x & y
    z.map(x => if (x) 1 else 0).reduce(_+_)
  }
  /**
    * Calculate the "true positive rate".
    * Here it is assumed that the class with label +1 is the class representing the "signal",
    * or a positive case in medicine.
    * How often has class 1 (with label +1 been correctly classified?)
    *
    * @param v0
    * @param labels
    * @return
    */
  def calcCorrectPredictionsClass1(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    val class1 = labels.map(x => x.toDouble).map(x=>if(x>0) 1 else 0).map(x => x.toDouble)
    (signum(v0) *:* class1).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  /**
    * Calculates the false positive rate.
    * @param v0
    * @param labels
    * @return
    */
  def calcInCorrectPredictionsClass1(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    //If 1: it is class 2 (label: -1) else the value is 0
    val class2 = labels.map(x => x.toDouble).map(x=>if(x<=0) 1 else 0).map(x => x.toDouble)
    (signum(v0) *:* class2).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  /**
    * How often has class 2 (with label -1 been correctly classified?)
    * @param v0
    * @param labels
    * @return
    */
  def calcCorrectPredictionsClass2(v0: DenseVector[Double], labels: DenseVector[Int]) : Int={
    assert(v0.length == labels.length)
    val class2 = labels.map(x => x.toDouble).map(x=>if(x<0) -1 else 0).map(x => x.toDouble)
    (signum(v0) *:* class2).map(x=>if(x>0) 1 else 0).reduce(_+_)
  }

  private def logClassDistribution (z: DenseVector[Double]) = {
    val positiveEntries = z.map(x => if (x > 0) 1 else 0).reduce(_ + _)
    val negativeEntries = z.map(x => if (x < 0) 1 else 0).reduce(_ + _)
    println("In vector with length: "+z.length+" there are: "+positiveEntries+" positive and: "+negativeEntries+" negative entries!")
  }
}