package dataframes

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

class DataframeLoader(val spark: SparkSession) {

  def getUsers(): DataFrame =
    spark.read
        .option("delimiter", ":")
        .csv("src/main/resources/data/users.dat")
        .toDF("UserID", "Gender", "Age", "Occupation" , "Zip-code")

  def getRatings(): DataFrame =
    spark.read
      .option("delimiter", ":")
      .csv("src/main/resources/data/ratings.dat")
      .toDF("UserID", "MovieID", "Rating", "Timestamp")
      .withColumn("Rating", col("Rating") / 5)

  def getMovies(): DataFrame =
    spark.read
      .option("delimiter", ":")
      .csv("src/main/resources/data/movies.dat")
      .toDF("MovieID", "Title" , "Genres");

}

object DataframeLoader {
  def apply(spark: SparkSession) = new DataframeLoader(spark)
}
