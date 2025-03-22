# Necessary imports.  This may need to be adjusted based on your
# specific Spark and system configuration.  This is a good starting point.
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier  # Added RandomForest
from pyspark.sql.functions import col, lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# --- 1. Data Loading and Preparation ---

def load_and_prepare_data(spark, mendeley_path, student_path):
    """
    Loads, merges, and prepares the Mendeley and Student datasets.

    Args:
        spark: SparkSession object.
        mendeley_path: Path to the Mendeley dataset.
        student_path: Path to the Student dataset.

    Returns:
        A tuple containing:
            - training_data: DataFrame for training.
            - test_data: DataFrame for testing.
            - merged_data: The combined dataset (for later retraining).
            - num_classes: The number of classes (for the final model).
    """

    # Load Mendeley dataset
    mendeley_df = spark.read.csv(mendeley_path, header=True, inferSchema=True)
    mendeley_df = mendeley_df.select("artist_name", "track_name", "release_date", "genre", "lyrics")
    mendeley_df = mendeley_df.withColumn("release_date", col("release_date").cast("int"))  #Ensure correct type
    mendeley_df = mendeley_df.na.drop()  # Drop rows with ANY missing values

    # Load Student dataset
    student_df = spark.read.csv(student_path, header=True, inferSchema=True)
    student_df = student_df.select("artist_name", "track_name", "release_date", "genre", "lyrics")
    student_df = student_df.withColumn("release_date", col("release_date").cast("int"))
    student_df = student_df.na.drop()

    # Merge datasets
    merged_data = mendeley_df.union(student_df)

    # Split Mendeley data (before merging for initial training)
    (training_mendeley, test_mendeley) = mendeley_df.randomSplit([0.8, 0.2], seed=12345)

    return training_mendeley, test_mendeley, merged_data, len(merged_data.select("genre").distinct().collect())

# --- 2. MLlib Pipeline Creation ---

def create_pipeline(num_classes):
    """
    Creates the MLlib pipeline.

    Args:
       num_classes: The number of output classes.

    Returns:
        The constructed MLlib pipeline.
    """
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
    # Use CountVectorizer instead of HashingTF
    cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize=10000, minDF=5.0)  # Adjust vocabSize and minDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    label_indexer = StringIndexer(inputCol="genre", outputCol="label").fit(merged_data) # Fit on the MERGED data!

    # Try different classifiers (Logistic Regression and RandomForest)
    # lr = LogisticRegression(maxIter=10, regParam=0.001)  # Initial settings, adjust these
    rf = RandomForestClassifier(numTrees=100, maxDepth=10, seed=42) # RandomForest example
    # You can swap between 'lr' and 'rf' here
    pipeline = Pipeline(stages=[tokenizer, cv, idf, label_indexer, rf])  # Use 'rf'
    return pipeline


# --- 3. Model Training and Evaluation ---

def train_and_evaluate(pipeline, training_data, test_data):
    """
    Trains the model and evaluates its performance.

    Args:
        pipeline: The MLlib pipeline.
        training_data: DataFrame for training.
        test_data: DataFrame for testing.

    Returns:
        The trained MLlib model.
    """
    model = pipeline.fit(training_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy}")

    evaluator = MulticlassClassificationEvaluator(
     labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions)
    print(f"F1-score: {f1}")

    return model


# --- 4. Main Execution (for run.bat) ---

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("MusicGenreClassification") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g")\
        .getOrCreate()


    # --- Initial Training (7 classes) ---
    mendeley_path = "data/Mendeley_dataset.csv"  # Path to the Mendeley dataset (replace with your actual path)
    student_path = "data/Student_dataset.csv"  # Path to your Student dataset
    training_data, test_data, merged_data, _ = load_and_prepare_data(spark, mendeley_path, student_path)

    # Get the number of classes BEFORE merging
    num_classes_initial = len(training_data.select("genre").distinct().collect())
    pipeline_initial = create_pipeline(num_classes_initial)
    model_initial = train_and_evaluate(pipeline_initial, training_data, test_data)
    model_initial.write().overwrite().save("initial_model")

    # --- Retraining (8 classes) ---
    num_classes_merged = len(merged_data.select("genre").distinct().collect())
    pipeline_merged = create_pipeline(num_classes_merged)
    model_merged = pipeline_merged.fit(merged_data)  # Train on the *entire* merged dataset.
    model_merged.write().overwrite().save("merged_model")


    spark.stop()