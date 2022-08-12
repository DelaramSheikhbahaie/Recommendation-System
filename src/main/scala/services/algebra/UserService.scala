package services.algebra

import models.UserDTO

trait UserService {

  def findById(id: Int): Option[UserDTO]
  def findByIds(ids: Seq[Int]): Seq[UserDTO]

}
