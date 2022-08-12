import dataframes.DataframeLoader
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types._
import services.algebra.{MovieService, UserService}
import services.impl.{MovieServiceImpl, UserServiceImpl}


object Main extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
//  System.setProperty("hadoop.home.dir", "C:\\Users\\lenovo\\hadoop")

  val spark = SparkSession.builder()
    .master("local[4]")
    .appName("recommenderV2")
    .getOrCreate()
// D:/Daneshgah/final project/code/recommenderV2/src/main/resources/data/data/users.dat


  val loader = DataframeLoader(spark)

  val users = loader.getUsers()
  val movies = loader.getMovies()
  val ratings = loader.getRatings()

  val userService: UserService = new UserServiceImpl(loader)
  val movieService: MovieService = new MovieServiceImpl(loader)
  userService.findById(1).foreach(println)
  movieService.findById(1).foreach(println)
  sys.exit(0)

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

  val getGenderInt = udf { (column: Option[String]) =>
    column match {
      case Some(value) if value equalsIgnoreCase "m" => 0
      case _ => 1
    }
  }
  val getGenreValue = udf { (oldStringGenres: String, genre: String) =>
    if (oldStringGenres.contains(genre)) 1 else 0
  }

  val moviesWithGenresFeatures = genres.foldLeft(movies) {
    case (updatingMovies, genre) =>
      updatingMovies.withColumn(genre, getGenreValue(updatingMovies.col("Genres"), lit(genre)))
  }

  val Data = ratings
    .join(users, ratings.col("UserID") === users.col("UserID"), "right")
    .join(moviesWithGenresFeatures, ratings.col("MovieID") === moviesWithGenresFeatures.col("MovieID"), "right")
    .select(
      users.col("UserID") cast IntegerType,
      moviesWithGenresFeatures.col("MovieID") cast IntegerType,
      ratings.col("Rating") cast DoubleType,
      getGenderInt(users.col("Gender")) as "Gender",
      users.col("Occupation") cast IntegerType,
      users.col("Age") cast IntegerType,
      when(ratings.col("Rating") gt 0.9, lit(1)).otherwise(lit(0)) as "label",
      moviesWithGenresFeatures.col("Adventure"),
      moviesWithGenresFeatures.col("Action"),
      moviesWithGenresFeatures.col("Animation"),
      moviesWithGenresFeatures.col("Children's"),
      moviesWithGenresFeatures.col("Comedy"),
      moviesWithGenresFeatures.col("Crime"),
      moviesWithGenresFeatures.col("Documentary"),
      moviesWithGenresFeatures.col("Drama"),
      moviesWithGenresFeatures.col("Fantasy"),
      moviesWithGenresFeatures.col("Film-Noir"),
      moviesWithGenresFeatures.col("Horror"),
      moviesWithGenresFeatures.col("Musical"),
      moviesWithGenresFeatures.col("Mystery"),
      moviesWithGenresFeatures.col("Romance"),
      moviesWithGenresFeatures.col("Sci-Fi"),
      moviesWithGenresFeatures.col("Thriller"),
      moviesWithGenresFeatures.col("War"),
      moviesWithGenresFeatures.col("Western")
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
    .setHandleInvalid("skip")

  val sizeHint = new VectorSizeHint()
    .setInputCol("features")
    .setHandleInvalid("skip")
    .setSize(4)

  val als = new ALS("")
    .setMaxIter(10)
    .setRegParam(0.3)
    .setRank(10)
    .setSeed(12345L)
    .setUserCol("UserID")
    .setItemCol("MovieID")
    .setRatingCol("Rating")
    .setPredictionCol("als_prediction")

  //creating pipeline
  val pipeline = new Pipeline().setStages(Array(assembler,  als))

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
