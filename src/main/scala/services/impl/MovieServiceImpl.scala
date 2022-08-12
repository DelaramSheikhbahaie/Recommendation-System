package services.impl

import dataframes.DataframeLoader
import models.MovieDTO
import org.apache.spark.sql.Row
import services.algebra.MovieService

import scala.util.Try

class MovieServiceImpl(val loader: DataframeLoader) extends MovieService {

  import MovieServiceImpl._
  override def findById(id: Int): Option[MovieDTO] =
    loader.getMovies()
      .filter { row =>
        row.getAs[String]("MovieID").toInt == id
      }
      .collect()
      .headOption
      .map(rowToMovie)

  override def findByIds(ids: Seq[Int]): Seq[MovieDTO] =
    loader.getMovies()
      .filter { row =>
        ids.contains(row.getAs[String]("MovieID").toInt)
      }
      .collect()
      .map(rowToMovie)

}

object MovieServiceImpl {
  private final val rowToMovie: Row => MovieDTO = { row =>
    val movieId = row.getAs[String]("MovieID").toInt
    val title = Try(row.getAs[String]("Title")).toOption
    val genres = Try(row.getAs[String]("Genres")).toOption.map(_.split("\\|").toSet).getOrElse(Set.empty[String])

    MovieDTO(
      id = movieId, title = title, genres = genres
    )
  }
}