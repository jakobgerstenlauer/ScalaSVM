package SVM

import breeze.linalg._
import breeze.numerics.signum
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

abstract class BaseMatrixFactory (d: Data, kf: KernelFunction, epsilon: Double) extends MatrixFactory {

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
    for (i <- 0 until N; j <- 0 until N){
      v(i) += z(j) * d.getLabelTrain(i) * kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
    }
    v
  }

  def predictOnTrainingSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N = d.getN_train
    val v = DenseVector.fill(N){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    for (i <- 0 until N; j <- 0 until N){
      v(i) += z(j) * kf.kernel(d.getRowTrain(i), d.getRowTrain(j))
    }
    signum(v)
  }

  def predictOnTestSet(alphas : DenseVector[Double]) : DenseVector[Double]  = {
    val N_train = d.getN_train
    val N_test = d.getN_test
    val v = DenseVector.fill(N_test){0.0}
    val z : DenseVector[Double] = alphas *:* d.getLabelsTrain.map(x=>x.toDouble)
    for (i <- 0 until N_test; j <- 0 until N_train){
      v(i) += z(j) * kf.kernel(d.getRowTrain(j), d.getRowTest(i))
    }
    signum(v)
  }

  override def getData (): Data = d
}

