import dataframes.DataframeLoader
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._


object Main extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir", "C:\\Users\\lenovo\\hadoop")

  val spark = SparkSession.builder()
    .master("local[4]")
    .appName("recommenderV2")
    .getOrCreate()
// D:/Daneshgah/final project/code/recommenderV2/src/main/resources/data/data/users.dat


  val loader = DataframeLoader(spark)

  val users = loader.getUsers()
  val movies = loader.getMovies()
  val ratings = loader.getRatings()


  val genres: Set[String] = Set(
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
  )

  val getMovieGenresFeature = udf { (column: String) =>
    val movieGenres = column.split("\\s*\\|\\s*").toSet
    val genresString = "1" + genres.map { genre =>
      if (movieGenres.contains(genre)) 1 else 0
    }.mkString("")

    genresString.toDouble
  }

  val likesOrNot = udf { (rating: Double) =>
    rating > 0.5
  }

  val getGenderInt = udf { (column: String) =>
    column match {
      case value if value equalsIgnoreCase "m" => 0
      case _ => 1
    }
  }
  val getGenreValue = udf { (oldStringGenres: String, genre: String) =>
    if (oldStringGenres.contains(genre)) 1 else 0
  }

  val getRatings = udf { (UserID: Double, MovieID: Double , RatingUserID: Double, RatingMovieID: Double , rating: Double ) =>
    if(UserID == RatingUserID && MovieID == RatingMovieID) Some(rating) else None
  }

  val MovieFeatures = genres.foldLeft(movies) {
    case (updatingMovies, genre) =>
      updatingMovies.withColumn(genre, getGenreValue(updatingMovies.col("Genres"), lit(genre)))
  }

  val userMovie = ratings
    .join(users)
    .join(MovieFeatures)
    .select(
      users.col("UserID") cast IntegerType as "UserID",
      MovieFeatures.col("MovieID") cast IntegerType as "MovieID",
    )

  val Data = ratings
    .join(users)
    .join(MovieFeatures)
    .join(userMovie)
    .select(
      userMovie.col("UserID") cast IntegerType as "UserID",
      userMovie.col("MovieID") cast IntegerType as "MovieID",
      getRatings(userMovie.col("UserID") , userMovie.col("MovieID") , ratings.col("UserID") , ratings.col("MovieID"), ratings.col("Rating") ) cast DoubleType as "Rating",
      getGenderInt(users.col("Gender")) as "Gender",
      users.col("Occupation") cast "Int" as "Occupation" ,
      users.col("Age") cast "Int" as "Age",
      likesOrNot(ratings.col("Rating")) cast IntegerType as "label",
      MovieFeatures.col("Adventure") cast "Int" as "Adventure",
      MovieFeatures.col("Action") cast "Int" as "Action",
      MovieFeatures.col("Animation") cast "Int" as "Animation",
      MovieFeatures.col("Children's") cast "Int" as "Children's",
      MovieFeatures.col("Comedy") cast "Int" as "Comedy",
      MovieFeatures.col("Crime") cast "Int" as "Crime",
      MovieFeatures.col("Documentary") cast "Int" as "Documentary",
      MovieFeatures.col("Drama") cast "Int" as "Drama",
      MovieFeatures.col("Fantasy") cast "Int" as "Fantasy",
      MovieFeatures.col("Film-Noir") cast "Int" as "Film-Noir",
      MovieFeatures.col("Horror") cast "Int" as "Horror",
      MovieFeatures.col("Musical") cast "Int" as "Musical",
      MovieFeatures.col("Mystery") cast "Int" as "Mystery",
      MovieFeatures.col("Romance") cast "Int" as "Romance",
      MovieFeatures.col("Sci-Fi") cast "Int" as "Sci-Fi",
      MovieFeatures.col("Thriller") cast "Int" as "Thriller",
      MovieFeatures.col("War") cast "Int" as "War",
      MovieFeatures.col("Western") cast "Int" as "Western",


    )

  Data.show(10,false)

  val Array(trainingData, testData) = Data.randomSplit(Array(0.8, 0.2))

  val assembler = new VectorAssembler()
    .setInputCols(Array(
      "Gender",
      "Age",
      "Occupation" ,
      "Adventure",
      "Action" ,
      "Animation"  ,
      "Children's" ,
      "Comedy" ,
      "Crime" ,
      "Documentary" ,
      "Drama" ,
      "Fantasy" ,
      "Film-Noir" ,
      "Horror" ,
      "Musical" ,
      "Mystery" ,
      "Romance" ,
      "Sci-Fi",
      "Thriller" ,
      "War",
      "Western"
    ))
    .setOutputCol("features")

  val sizeHint = new VectorSizeHint()
    .setInputCol("features")
    .setHandleInvalid("skip")
    .setSize(4)

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFeaturesCol("features")   // setting features column
    .setLabelCol("label")       // setting label column

  //creating pipeline
  val pipeline = new Pipeline().setStages(Array(assembler,  lr))

  //fitting the model

  val lrModel = pipeline.fit(trainingData)

  val result = lrModel.transform(testData)

  val predictionAndLabels = result.select("Rating", "features", "label")
  val evaluator = new MulticlassClassificationEvaluator()
    .setPredictionCol("Rating")
    .setMetricName("accuracy")

  result.show(100, false)

  println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

  System.in.read()

}
