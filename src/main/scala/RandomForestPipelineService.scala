import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object RandomForestPipelineService extends App {

  val spark = SparkSession.builder()
    .appName("ML pipeline for Iris classification")
    .config("spark.master", "local")
    .getOrCreate()

  val data = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("src/main/resources/IRIS.csv")

  val assembler = new VectorAssembler()
    .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
    .setOutputCol("features")

  private val speciesIndexer = new StringIndexer()
    .setInputCol("species")
    .setOutputCol("label")
    .fit(data)

  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  val randomForestClassifier = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(10)


  val labels = Array("Iris-setosa", "Iris-versicolor", "Iris-virginica")
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labels)

  val pipeline = new Pipeline()
    .setStages(Array(speciesIndexer, assembler, randomForestClassifier, labelConverter))

  val model = pipeline.fit(trainingData)
  model.write.overwrite().save("model")

  val predictions = model.transform(testData.drop("species"))

  predictions.select(col("sepal_length"), col("sepal_width"),
    col("petal_length"), col("petal_width"), col("prediction"))
    .show()

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test Error = ${(1.0 - accuracy)}")

  val randomForestClassificationModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println(s"Learned classification forest model:\n ${randomForestClassificationModel.toDebugString}")
}
