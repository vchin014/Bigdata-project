package edu.ucr.cs.cs226.vchin014
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author ${Vishal Rohith Chinnam}
 */

object DataFrameTransformation {
  def transformation(spark: SparkSession, inputDF: DataFrame, outputPath: String){
    import spark.implicits._
    import org.apache.spark.sql.functions.to_date
    import org.apache.spark.sql.functions.hour
    import org.apache.spark.sql.functions.minute
    import org.apache.spark.sql.functions.second
    import org.apache.spark.sql.functions.month
    import org.apache.spark.sql.functions.dayofmonth
    import org.apache.spark.sql.functions.year


    val df2 = inputDF.toDF(inputDF.columns: _*) // Loading cleaned data here

    // Splitting 'Start_Time' column
    val df3 = df2
      .withColumn("Start_Date", to_date($"Start_Time"))
      .withColumn("Start_Hour", hour($"Start_Time"))
      .withColumn("Start_Mins", minute($"Start_Time"))
      .withColumn("Start_seconds", second($"Start_Time"))

    // Splitting 'Weather_Timestamp' column
    val df4 = df3
      .withColumn("Weather_Date", to_date($"Weather_Timestamp"))
      .withColumn("Weather_Hour", hour($"Weather_Timestamp"))
      .withColumn("Weather_Mins", minute($"Weather_Timestamp"))
      .withColumn("Weather_seconds", second($"Weather_Timestamp"))

    // Extracting 'Start_month', 'Start_day', 'Start_year'
    val df5 = df4
      .withColumn("Start_month", month($"Start_Date"))
      .withColumn("Start_day", dayofmonth($"Start_Date"))
      .withColumn("Start_year", year($"Start_Date"))

    // Extracting 'Weather_month', 'Weather_day', 'Weather_year'
    val df6 = df5
      .withColumn("Weather_month", month($"Weather_Date"))
      .withColumn("Weather_day", dayofmonth($"Weather_Date"))
      .withColumn("Weather_year", year($"Weather_Date"))

    //dropping Start_Time and Weather_Timestamp columns
    val df7 = df6.drop("Start_Time", "Weather_Timestamp")

    // Show the content of df6
    df7.show()

    //getting an optimal partition count
    val optimalPartitionCount = Math.ceil(df7.count() / (3 * 1024 * 1024)).toInt // Assuming 3GB data size

    // Using the calculated optimal partition count for repartitioning
    val repartitionedDF = df7.repartition(optimalPartitionCount)

    // Writing to Parquet
    repartitionedDF.write.mode("overwrite").parquet(outputPath)

  }

}
