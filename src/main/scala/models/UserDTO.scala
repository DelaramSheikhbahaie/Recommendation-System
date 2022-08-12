package models

// "UserID", "Gender", "Age", "Occupation" , "Zip-code"
case class UserDTO(
                  id: Long,
                  gender: Option[String],
                  age: Option[Int],
                  occupation: Option[Int],
                  zipCode: Option[Int]
                  )
