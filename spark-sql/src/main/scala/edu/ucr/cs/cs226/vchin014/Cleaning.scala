package edu.ucr.cs.cs226.vchin014
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * @author ${Vishal Rohith Chinnam}
 */

    object Cleaning {
    def cleanData(spark: SparkSession, inputPath: String, outputPath: String) {

      import org.apache.log4j.{Level, Logger}
      import org.apache.spark.sql.functions.{col, when}
      import org.apache.spark.sql.expressions.Window
      import org.apache.spark.sql.functions.{coalesce, col}
      Logger.getRootLogger.setLevel(Level.WARN)
      import org.apache.spark.sql.DataFrame
      import org.apache.spark.sql.functions._


      // Read CSV using Spark SQL
      val spark_df: DataFrame = spark.read.option("header", "true").option("inferSchema", "true").csv(inputPath)

      spark_df.cache()

      //Data Preprocessing
      //Data cleaning

      // Drop the number of duplicate rows
      val duplicateCount = spark_df.dropDuplicates().count()

      // Print the count of duplicate rows
      println(s"Number of duplicate rows: $duplicateCount")


      import org.apache.spark.sql.functions.col
      import org.apache.spark.sql.functions._
      import org.apache.spark.sql.expressions.Window

      // Data cleaning: Dropping rows with at least 40 NaN values
      val threshold = 40

      val dfFiltered = spark_df.na.drop(threshold, spark_df.columns)

      //Filtering the columns that aren't required post data analysis
      val selectedColumns = Seq(
        "ID", "Severity", "Start_Time", "Start_Lat", "Start_Lng", "Street", "City", "State", "Zipcode",
        "Timezone", "Weather_Timestamp", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition",
        "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout",
        "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
      )

      val dfSelected = dfFiltered.select(selectedColumns.map(col): _*)

      val limit = 5

      // Applying forward fill and backward fill for selected columns
      val filledDF = selectedColumns.foldLeft(dfSelected) { (tempDF, colName) =>
        // Defining forward fill expression
        val ffillExpr = org.apache.spark.sql.functions.when(
          col(colName).isNotNull,
          col(colName)
        ).otherwise(
          org.apache.spark.sql.functions.lag(col(colName), limit).over(Window.orderBy("ID"))
        )

        // Defining backward fill expression
        val bfillExpr = org.apache.spark.sql.functions.when(
          col(colName).isNotNull,
          col(colName)
        ).otherwise(
          org.apache.spark.sql.functions.lead(col(colName), limit).over(Window.orderBy("ID"))
        )

        // Applying forward fill and backward fill to the column
        tempDF.withColumn(colName, coalesce(ffillExpr, bfillExpr))
            }


      // selecting columns with more null values
      val columnsToDrop = Seq("Wind_Chill(F)", "Wind_Speed(mph)", "Precipitation(in)")

      // Dropping rows with missing values in the above specified columns
      val dfWithoutMissingValues = filledDF.na.drop(columnsToDrop)

      // Calculating the optimal partition count based on desired partition size
      val optimalPartitionCount = Math.ceil(dfWithoutMissingValues.count() / (3 * 1024 * 1024)).toInt  // Assuming 3GB data size

      // Using the calculated optimal partition count for repartitioning
      val repartitionedDF = dfWithoutMissingValues.repartition(optimalPartitionCount)

      // Writing to Parquet
      repartitionedDF.write.mode("overwrite").parquet(outputPath)

      }
    }
