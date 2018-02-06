/*
package SVM

trait FlyWeightFactory [T1,T2] extends Function [T1,T2] {
  private var pool = Map[T1,T2]()
  def createFlyWeight(intrinsic:T1):T2
  def apply (index : T1 ) : T2 = {
    pool.get(index) match {
      case Some( f ) => f
      case None => {
        pool += (index −> createFlyWeight(index))
        pool (index)
      }
    }
  }
  def update (index:T1, elem:T2){
    pool(index)=elem
  }
}
*/
