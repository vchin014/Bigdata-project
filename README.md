# CAR ACCIDENTS PREVENTION USING  SPARK & SPARK.ML

## Directory Structure

You are currently in the main directory `spark-sql/`.

## Dependencies Installation

Ensure all dependencies listed in the `pom.xml` file are installed. If not, add them and reload the Maven project. Then execute the following Maven command:

```bash
mvn clean install
```

## Data Setup

Place the original CSV data in the following location:

```
data/input/original_data.csv
```

## Data Cleaning

To clean the data, follow these steps:

1. Open the file `src/main/scala/edu/ucr/cs/cs226/vchin014/App.scala`.
2. Uncomment the following line:

   ```scala
   Cleaning.cleanData(spark, inputPath, cleanedDataPath)
   ```

## Data Transformation

To transform the cleaned data before applying machine learning algorithms, perform the following:

1. In `App.scala`, uncomment the following line:

   ```scala
   DataFrameTransformation.transformation(spark, cleanedDF, dataFrameTransformationPath)
   ```

## ML Algorithm Application

Apply machine learning algorithms to the cleaned and transformed data by following these steps:

- **RandomForest Classifier:**  
  Uncomment the following line in `App.scala`:

  ```scala
  codeAlgorithms.applyAlgorithms(spark, transformedDF)
  ```

- **GBT Regression:**  
  Uncomment the following line:

  ```scala
  GbtAlgorithm.applyAlgorithms(spark, transformedDF)
  ```

- **PCA-Linear Regression:**  
  Uncomment the following line:

  ```scala
  PcaAlgorithm.applyAlgorithms(spark, transformedDF)
  ```

## Execution

### Building the JAR

Build and package your code into a JAR file (`spark-sql-1.0-SNAPSHOT.jar`) using the following Maven command:

```bash
mvn package
```

### Running the Application

Run the application using the `spark-submit` command as shown below:

```bash
spark-submit \
  --master "local[*]" \
  --executor-memory 16G \
  --driver-memory 10G \
  --conf spark.driver.memoryOverhead=4G \
  --conf spark.executor.memoryOverhead=4G \
  --conf spark.driver.maxResultSize=8G \
  --conf spark.sql.shuffle.partitions=32 \
  --conf spark.default.parallelism=32 \
  --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=2" \
  target/spark-sql-1.0-SNAPSHOT.jar
```

**Note:** Adjust the `spark-submit` parameters according to your system configuration as the execution is on a local system.
