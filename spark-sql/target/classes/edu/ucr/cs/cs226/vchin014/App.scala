/**
 *
 * Author Contributions:
 * - Vishal Rohith Chinnam: Implemented the initial data cleaning and preprocessing (App.scala, Cleaning.scala).
 * - Aswini Shilpha S S: Performed Data analysis and visualizations on the cleaned data (Analysis.scala).
 * - Vishal Rohith Chinnam: Implemented the feature engineering and data transformation required to implement ml models(DataFrameTransformation.scala)
 * - Vishal Rohith Chinnam: Implemented the ml models RandomForestClassifier, GradientBoostedTreesRegressor and PCA-LinearRegression models(Algorithms.scala, GbtAlgorithm.scala, PcaAlgorithm.scala)
 * - Vishal Rohith Chinnam: Worked on evaluation metrics and result analysis(Algorithms.scala, GbtAlgorithm.scala, PcaAlgorithm.scala)
 * - Vishal Rohith Chinnam: Implemented parquet read and write post cleaning and as well as transforming the data
 */



package edu.ucr.cs.cs226.vchin014

import org.apache.spark.sql.DataFrame

/**
 * @author ${Vishal Rohith Chinnam}
 */
object App {


  def main(args : Array[String]) {
    import org.apache.spark.SparkConf
    import org.apache.spark.sql.SparkSession
    import org.apache.log4j.{Level, Logger}
    Logger.getRootLogger.setLevel(Level.WARN)
    import org.apache.spark.sql.DataFrame
    import org.apache.spark.sql.functions._

    val conf = new SparkConf().setMaster("local")
    val spark = SparkSession
      .builder()
      .appName("BigData SparkSQL Demo")
      .config(conf)
      .getOrCreate()

    // Specifying input and output paths
    //val inputPath = "/Users/vishal/Workspace/spark-sql/data/input/original_data.csv" //original data path
    //val cleanedDataPath = "/Users/vishal/Workspace/spark-sql/data/input/cleaned_data.parquet" //cleaned data path
    val dataFrameTransformationPath = "/Users/vishal/Workspace/spark-sql/data/input/transformed_data.parquet" //transformed data path

    // Uncomment the line below if you need to re-run the cleaning process
    //Cleaning.cleanData(spark, inputPath, cleanedDataPath)

    // Loading the cleaned data
    //val cleanedDF = spark.read.parquet(cleanedDataPath)

    //Loading the transformed data
    val transformedDF = spark.read.parquet(dataFrameTransformationPath)

    //Uncomment the line below if you need to re-transform the cleaned data
    //DataFrameTransformation.transformation(spark, cleanedDF, dataFrameTransformationPath)

    //uncomment to apply RandomForest Classifier ml algorithm
    //Algorithms.applyAlgorithms(spark, transformedDF)

    //applying GBT Regression ml algorithm
    GbtAlgorithm.applyAlgorithms(spark, transformedDF)

    //uncomment to apply PCA-linear_regression ml algorithm
    //PcaAlgorithm.applyAlgorithms(spark, transformedDF)



    // Stop SparkSession
    spark.stop()


  }

}
