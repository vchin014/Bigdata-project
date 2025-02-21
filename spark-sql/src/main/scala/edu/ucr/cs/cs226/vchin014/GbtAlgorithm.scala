package edu.ucr.cs.cs226.vchin014
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author ${Vishal Rohith Chinnam}
 */

object GbtAlgorithm {
  def applyAlgorithms(spark: SparkSession, inputDF: DataFrame){

    import org.apache.spark.ml.regression.{GBTRegressor, GBTRegressionModel}
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions
    import org.apache.spark.sql.types.DoubleType
    import org.apache.spark.sql.functions.col
    import org.apache.log4j.{Level, Logger}
    Logger.getRootLogger.setLevel(Level.WARN)

    // Ensuring 'Severity' is of type Double
    val dfWithDoubleSeverity = inputDF.withColumn("Severity", col("Severity").cast(DoubleType))

    //sampling the data
    val sampledDf = dfWithDoubleSeverity.sample(false, 0.5, 42).na.drop()

    // Features collection
    val features = Array("Start_Lat", "Start_Lng", "Temperature(F)", "Wind_Chill(F)",
      "Visibility(mi)", "Humidity(%)", "Pressure(in)", "Start_Hour", "Weather_Hour")

    // Assembling features
    val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
    val assembledDF = assembler.transform(sampledDf)

    // Splitting the data into train and test data
    val Array(train, test) = assembledDF.randomSplit(Array(0.7, 0.3), seed = 42)

    // Assigning the model
    val xgb = new GBTRegressor().setLabelCol("Severity").setFeaturesCol("features").setMaxIter(100).setSeed(42)

    // Fitting the model
    val model = xgb.fit(train)

    // Evaluating the model
    val predictions = model.transform(test)
    predictions.show()

    // Evaluating model using RMSE and MAE
    val rmse = math.sqrt(predictions.select("Severity", "prediction").rdd.map(row =>
      math.pow(row.getDouble(0) - row.getDouble(1), 2)).mean())

    val mae = predictions
      .select(functions.abs(functions.col("Severity") - functions.col("prediction")).alias("mae"))
      .agg("mae" -> "avg")
      .first()
      .getDouble(0)

    println(s"RMSE: $rmse")
    println(s"MAE: $mae")

  }

}
