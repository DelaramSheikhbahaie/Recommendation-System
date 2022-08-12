package services.algebra

import models.MovieDTO

trait MovieService {

  def findById(id: Int): Option[MovieDTO]
  def findByIds(ids: Seq[Int]): Seq[MovieDTO]

}
