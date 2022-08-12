package services.impl

import dataframes.DataframeLoader
import models.UserDTO
import org.apache.spark.sql.Row
import services.algebra.UserService

import scala.util.Try

class UserServiceImpl(val loader: DataframeLoader) extends UserService {

  import UserServiceImpl._

  override def findById(id: Int): Option[UserDTO] =
    loader.getUsers()
      .filter { row =>
        row.getAs[String]("UserID").toInt == id
      }
      .collect()
      .headOption
      .map(rowToUser)

  override def findByIds(ids: Seq[Int]): Seq[UserDTO] =
    loader.getUsers()
      .filter(row => ids.contains(row.getAs[String]("UserID").toInt))
      .collect()
      .map(rowToUser)

}

object UserServiceImpl {
  private final val rowToUser: Row => UserDTO = { row =>
    val userId = row.getAs[String]("UserID")
    val gender = Option(row.getAs[String]("Gender"))
    val age = Option(row.getAs[String]("Age"))
    val occupation = Option(row.getAs[String]("Occupation"))
    val zipCode = Option(row.getAs[String]("Zip-code"))

    UserDTO(
      userId.toLong,
      gender,
      age.flatMap(Try(_).map(_.toInt).toOption),
      occupation.flatMap(Try(_).map(_.toInt).toOption),
      zipCode.flatMap(Try(_).map(_.toInt).toOption)
    )
  }
}
