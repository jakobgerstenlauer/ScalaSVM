import org.scalatest.FunSuite
import SVM.Alphas
import SVM.Alphas
import breeze.linalg._

class AlphasTest extends FunSuite{

  test("Median must be 5.5."){	
	val N = 10
	val alpha : DenseVector[Double] = new DenseVector(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
	val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
	val alphas = Alphas(N, alpha, alpha_old)
	assert(alphas.getQuantile(0.5) == 5.5)
  }
}
