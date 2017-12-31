package SVM

import breeze.linalg._
import breeze.numerics.signum

import scala.collection.mutable.{Set=>MSet, HashMap, MultiMap}

abstract class BaseMatrixFactory (d: Data, kf: KernelFunction, epsilon: Double) extends MatrixFactory {

  /**
    * key: row index of matrix K
    * value: set of non-sparse column indices of matrix K
    */
  val rowColumnPairs : MultiMap[Int, Int] = initializeRowColumnPairs();

  /**
    * Check if the similarity between instance i and j is significant.
    * @param i row index of K
    * @param j column index of K
    * @return
    */
  def entryExists(i: Int, j: Int):Boolean = {
    rowColumnPairs.entryExists(i, _ == j)
  }

  /**
    * Check if the similarity between test instance i and training instance j is significant.
    * @param i row index of S
    * @param j column index of S
    * @return
    */
  def entryExistsTest(i: Int, j: Int):Boolean = {
    rowColumnPairsTest.entryExists(i, _ == j)
  }

  /**
    * key: row index of matrix S (index of test instance)
    * value: set of non-sparse column indices of matrix S
    */
  val rowColumnPairsTest : MultiMap[Int, Int] = initializeRowColumnPairsTest();

  def initializeRowColumnPairs(): MultiMap[Int, Int] = {
    val map: MultiMap[Int, Int] = new HashMap[Int, MSet[Int]] with MultiMap[Int, Int]
    val N = d.getN_train
    for (i <- 0 until N; j <- 0 until N;
         if(kf.kernel(d.getRowTrain(i), d.getRowTrain(j)) > epsilon)){
       map.addBinding(i,j)
    }
    map
  }

  def initializeRowColumnPairsTest(): MultiMap[Int, Int] = {
    val map: MultiMap[Int, Int] = new HashMap[Int, MSet[Int]] with MultiMap[Int, Int]
    val N_train = d.getN_train
    val N_test = d.getN_test
    for (i <- 0 until N_test; j <- 0 until N_train;
         if(kf.kernel(d.getRowTest(i), d.getRowTrain(j)) > epsilon)){
      map.addBinding(i,j)
    }
    map
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
  def calculateGradient(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N = d.getN_train
    val v = DenseVector.fill(N){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    for ((i,set) <- rowColumnPairs; j <- set){
      v(i) += z(j) * d.getLabelTrain(i) * kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
    }
    v
  }

  def predictOnTrainingSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N = d.getN_train
    val v = DenseVector.fill(N){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    for ((i,set) <- rowColumnPairs; j <- set){
      v(i) += z(j) * kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
    }
    signum(v)
  }

  def predictOnTestSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N_test = d.getN_test
    val v = DenseVector.fill(N_test){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    for ((i,set) <- rowColumnPairsTest; j <- set){
      v(i) += z(j) * kf.kernel( d.getRowTest(i), d.getRowTrain(j))
    }
    signum(v)
  }

  override def getData (): Data = d
}

