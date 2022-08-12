package models

case class MovieDTO(
                   id: Int,
                   title: Option[String],
                   genres: Set[String]
                   )
