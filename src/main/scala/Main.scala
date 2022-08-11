import dataframes.DataframeLoader
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object Main extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir", "C:\\Users\\lenovo\\hadoop")

  val spark = SparkSession.builder()
    .master("local[4]")
    .appName("recommenderV2")
    .getOrCreate()

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
      users.col("UserID"),
      moviesWithGenresFeatures.col("MovieID"),
      ratings.col("Rating"),
      getGenderInt(users.col("Gender")) as "Gender",
      users.col("Occupation") cast IntegerType,
      users.col("Age") cast IntegerType,
      when(ratings.col("Rating") gt 0.74, lit(1)).otherwise(lit(0)) cast DoubleType as "label",
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
  val model = ALS.train(ratings.rdd.asInstanceOf[RDD[Rating]], 0, 0)

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

//  val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
//    val prediction = model.predict(features)
//    (prediction, label)
//  }

  val predictionAndLabels = result.select("Rating", "features", "label")
  val evaluator = new MulticlassClassificationEvaluator()
    .setPredictionCol("label")
    .setMetricName("accuracy")

  result.show(100, false)

  println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

  System.in.read()

}
