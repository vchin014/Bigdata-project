package edu.ucr.cs.cs226.vchin014
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author ${Vishal Rohith Chinnam}
 */

object Algorithms {
  def applyAlgorithms(spark: SparkSession, inputDF: DataFrame) {
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
    import org.apache.spark.ml.classification.RandomForestClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.sql.{Encoder, Encoders}
    import org.apache.spark.sql.types.{StringType, BooleanType}
    import org.apache.spark.sql.functions.col
    import spark.implicits._
    import org.apache.spark.mllib.evaluation.MulticlassMetrics


    // sampling the input data
    val sampledDf = inputDF.sample(false, 0.01, 42).na.drop()

    // Extract categorical columns
    val categoricalCols = sampledDf.schema.fields.filter(field =>
      field.dataType.isInstanceOf[StringType] ||
        field.dataType.isInstanceOf[BooleanType]
    ).map(_.name)

    // Convert string columns to numerical using StringIndexer
    val stringIndexers = categoricalCols.filter(c => sampledDf.schema(c).dataType.isInstanceOf[StringType])
      .map(col => new StringIndexer().setInputCol(col).setOutputCol(s"${col}_index"))

    // One-hot encode the indexed string columns
    val oneHotEncoders = stringIndexers.map(si => new OneHotEncoder().setInputCol(si.getOutputCol).setOutputCol(s"${si.getOutputCol}_onehot"))

    // Identifying boolean columns
    val booleanCols = categoricalCols.filter(c => sampledDf.schema(c).dataType.isInstanceOf[BooleanType])

    // Identifying boolean columns
    val numericCols = sampledDf.schema.fields.filter(field =>
      field.dataType.isInstanceOf[org.apache.spark.sql.types.NumericType]
    ).map(_.name)

    // Assemble features including one-hot encoded columns, boolean and numeric columns
    val assembler = new VectorAssembler()
      .setInputCols((oneHotEncoders.map(_.getOutputCol) ++ booleanCols ++ numericCols).toArray)
      .setOutputCol("features")

    // Initializing a RandomForestClassifier
    val rfClassifier = new RandomForestClassifier()
      .setLabelCol("Severity")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    // Creating a pipeline with string indexers, one-hot encoders, assembler, and any additional stages
    val pipeline = new Pipeline().setStages(stringIndexers ++ oneHotEncoders ++ Array(assembler, rfClassifier))
    // Fit the pipeline to the data
    val model = pipeline.fit(sampledDf)

    // Printing the schema of the DataFrame after transformations
    model.transform(sampledDf).printSchema()

    // Splitting the data into training and testing sets
    val Array(train, test) = sampledDf.randomSplit(Array(0.7, 0.3), seed = 42)

    // Making predictions on the test set
    val predictions = model.transform(test)

    predictions.show()  // Display the DataFrame to see predicted column names


    // Evaluating the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Severity")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Evaluating the accuracy
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: ${accuracy * 100}%%")

    // Evaluating the precision, recall and F-1 score
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = evaluator.setMetricName("f1").evaluate(predictions)

    println(s"Precision: $precision")
    println(s"Recall: $recall")
    println(s"F1 Score: $f1Score")


  }


}

