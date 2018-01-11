import org.scalatest.FunSuite
import SVM.Alphas
import SVM.Alphas
import breeze.linalg._
import breeze.numerics.{pow, sqrt}

class AlphasTest extends FunSuite{

  test("Momentum must be 1/3."){
    val alpha : DenseVector[Double] = new DenseVector(Array(1.0,2.0,3.0))
    val alpha_old : DenseVector[Double] = new DenseVector(Array(2.0,2.0,2.0))
    val alphas = Alphas(3, alpha, alpha_old)
    val expectedValue : DenseVector[Double] = alpha + alpha_old / 3.0
    val diff : DenseVector[Double] = alphas.updateAlphaAsConjugateGradient().alpha - expectedValue
    val distance : Double = sqrt(diff.map(x => pow(x,2)).reduce(_ + _))
    val epsilon = 0.001
    assert(distance < epsilon)
  }

  test("Conjugate gradient must be identical to original alpha if norm of alpha old is zero."){
    val N = 10
    val alpha : DenseVector[Double] = new DenseVector(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
    val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
    val alphas = Alphas(N, alpha, alpha_old)
    assert(alphas.updateAlphaAsConjugateGradient().alpha == alpha)
  }

  test("Conjugate gradient must be identical to original alpha if alpha old is identical to new alpha."){
    val N = 10
    val vector = new DenseVector(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
    val alpha : DenseVector[Double] = vector
    val alpha_old : DenseVector[Double] = vector
    val alphas = Alphas(N, alpha, alpha_old)
    assert(alphas.updateAlphaAsConjugateGradient().alpha == alpha)
  }

  test("After clipping 50% quantile, sparsity must be 50%."){
    val N = 10
    val alpha : DenseVector[Double] = new DenseVector(Array(2.0,1.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0))
    val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
    val alphas = Alphas(N, alpha, alpha_old)
    val alphasClipped = alphas.clipAlphas(0.5)
    val sumNonSparse = alphasClipped.alpha.map(x=>if(x>0) 1 else 0 ).reduce(_ + _)
    assert(sumNonSparse == 5)
  }

  test("getDelta() must return 10."){
    val N = 10
    val alpha : DenseVector[Double] = DenseVector.fill(N){1.0}
    val alpha_old : DenseVector[Double] = DenseVector.fill(N){0.0}
    val alphas = Alphas(N, alpha, alpha_old)
    assert(alphas.getDelta == 10.0)
  }

  test("Mean must be 5.0."){
    assert(Alphas.mean(0.0,10.0) == 5.0)
  }

  test("First element in sorted array must be 1 and the original vector must not change."){
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
