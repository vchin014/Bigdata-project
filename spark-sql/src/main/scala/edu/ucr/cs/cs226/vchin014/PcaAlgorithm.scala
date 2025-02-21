package edu.ucr.cs.cs226.vchin014
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}


/**
 * @author ${Vishal Rohith Chinnam}
 */

object PcaAlgorithm {
  def applyAlgorithms(spark: SparkSession, inputDF: DataFrame) {

    import org.apache.spark.ml.{Pipeline, PipelineModel}
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.regression.LinearRegression
    import org.apache.spark.ml.feature.PCA
    import org.apache.spark.sql.functions

    // Ensuring 'Severity' is of type Double
    val dfWithDoubleSeverity = inputDF.withColumn("Severity", col("Severity").cast(DoubleType))

    // Sampling data
    val sampledDf = dfWithDoubleSeverity.sample(false, 0.5, 42).na.drop()

    // Features collection
    val features = Array("Start_Lat", "Start_Lng", "Temperature(F)", "Wind_Chill(F)",
      "Visibility(mi)", "Humidity(%)", "Pressure(in)", "Start_Hour", "Weather_Hour")

    // Assemble features
    val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
    val assembledDF = assembler.transform(sampledDf)

    // Setting Target variable
    val target = "Severity"

    // Splitting data into train and test data
    val Array(train, test) = assembledDF.randomSplit(Array(0.7, 0.3), seed = 42)

    // Applying PCA
    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(features.length)

    //Applying Linear Regression model
    val lr = new LinearRegression().setLabelCol(target).setFeaturesCol("pcaFeatures")

    // Building a pipeline
    val pipeline = new Pipeline().setStages(Array(pca, lr))

    // Fitting the pipeline
    val model = pipeline.fit(train)

    // Transform test data
    val pcaTest = model.transform(test)

    // Evaluating the model
    val predictions = pcaTest.select(target, "prediction")
    predictions.show()

    // Evaluation using metrics RMSE and MAE
    val rmse = math.sqrt(predictions.rdd.map(row =>
      math.pow(row.getDouble(0) - row.getDouble(1), 2)).mean())

    val mae = predictions
      .select(functions.abs(functions.col(target) - functions.col("prediction")).alias("mae"))
      .agg("mae" -> "avg")
      .first()
      .getDouble(0)

    println(s"RMSE: $rmse")
    println(s"MAE: $mae")

  }

}
