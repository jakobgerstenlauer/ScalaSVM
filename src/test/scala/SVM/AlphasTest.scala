import org.scalatest.FunSuite
import SVM.Alphas
import SVM.Alphas
import breeze.linalg._

class AlphasTest extends FunSuite{

  test("First element in sorted array must be 1."){
    val N = 10
    val alpha : DenseVector[Double] = new DenseVector(Array(2.0,1.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
    val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
    val alphas = Alphas(N, alpha, alpha_old)
    val sorted = alphas.getSortedAlphas
    //the returned array must contain 1.0 at position 0!
    assert(sorted(0) == 1.0)
    //the original vector must not be changed!
    assert(alphas.alpha(0) == 2.0)
  }

  test("Median must be 5.5."){	
	val N = 10
	val alpha : DenseVector[Double] = new DenseVector(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
	val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
	val alphas = Alphas(N, alpha, alpha_old)
	assert(alphas.getQuantile(0.5) == 5.5)
  }

  test("0% Quantile must be 1."){
        val N = 10
        val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
        val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
        val alphas = Alphas(N, alpha, alpha_old)
        assert(alphas.getQuantile(0.0) == 1.0)
  }

  test("1% Quantile must be 1."){
        val N = 10
        val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
        val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
        val alphas = Alphas(N, alpha, alpha_old)
        assert(alphas.getQuantile(0.01) == 1.0)
  }

  test("10% Quantile must be 1."){
        val N = 10
        val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
        val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
        val alphas = Alphas(N, alpha, alpha_old)
        assert(alphas.getQuantile(0.1) == 1.0)
  }

  test("90% Quantile must be 1."){
        val N = 10
        val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
        val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
        val alphas = Alphas(N, alpha, alpha_old)
        assert(alphas.getQuantile(0.9) == 1.0)
  }

  test("100% Quantile must be 1."){
        val N = 10
        val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
        val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
        val alphas = Alphas(N, alpha, alpha_old)
        assert(alphas.getQuantile(1.0) == 1.0)
  }
}
