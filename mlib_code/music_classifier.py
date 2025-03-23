import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StringIndexer, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, when, length
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class MusicGenreClassifier:
    """A class to handle music genre classification using PySpark."""

    def __init__(self, spark_session=None, app_name="MusicGenreClassification"):
        """Initialize the classifier with a SparkSession."""
        self.spark = spark_session or self._create_spark_session(app_name)
        self.model = None
        self.pipeline = None
        self.label_indexer = None
        self.num_classes = None
        self.label_converter = None

    def _create_spark_session(self, app_name):
        """Create and configure a SparkSession."""
        return (SparkSession.builder
                .appName(app_name)
                .master("local[*]")
                .config("spark.driver.memory", "8g")
                .config("spark.executor.memory", "8g")
                .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
                .config("spark.sql.shuffle.partitions", "100")  # Better for large datasets
                .config("spark.default.parallelism", "100")  # Better parallelism
                .getOrCreate())

    def load_data(self, file_path, required_columns=None):
        """
        Load data from a CSV file and perform basic validation.

        Args:
            file_path: Path to the CSV file
            required_columns: List of required columns (if None, no validation)

        Returns:
            DataFrame if successful, None if file doesn't exist
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        try:
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)

            # Validate required columns if specified
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"Missing required columns: {missing_cols}")
                    return None

            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None

    def prepare_data(self, df, required_columns=None):
        """
        Clean and prepare the data for modeling.

        Args:
            df: Input DataFrame
            required_columns: List of columns to select

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            return None

        try:
            # Select required columns if specified
            if required_columns:
                df = df.select(*required_columns)

            # Convert release_date to integer
            df = df.withColumn("release_date", col("release_date").cast("int"))

            # Filter out records with empty lyrics
            df = df.filter(length(col("lyrics")) > 10)

            # Handle nulls more carefully (could be replaced with imputation)
            df = df.na.drop()

            return df
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def create_pipeline(self, training_data, hyperparameter_tuning=False):
        """
        Create the ML pipeline with text processing and classification.

        Args:
            training_data: DataFrame for fitting StringIndexer
            hyperparameter_tuning: Whether to enable hyperparameter tuning

        Returns:
            Pipeline object or CrossValidator if hyperparameter_tuning is True
        """
        # Create label indexer for genre
        self.label_indexer = StringIndexer(inputCol="genre", outputCol="label")

        # Text processing pipeline
        tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
        cv = CountVectorizer(inputCol="words", outputCol="rawFeatures",
                             vocabSize=10000, minDF=5.0)
        idf = IDF(inputCol="rawFeatures", outputCol="features")

        # Add release date as a feature
        # year_assembler = VectorAssembler(
        #    inputCols=["release_date"], outputCol="yearFeature")

        # Create classifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                    numTrees=100, maxDepth=10, seed=42)

        # Create the pipeline
        stages = [tokenizer, cv, idf, self.label_indexer, rf]
        self.pipeline = Pipeline(stages=stages)

        if not hyperparameter_tuning:
            return self.pipeline

        # Create parameter grid for tuning
        paramGrid = (ParamGridBuilder()
                     .addGrid(cv.vocabSize, [5000, 10000])
                     .addGrid(rf.numTrees, [50, 100])
                     .addGrid(rf.maxDepth, [5, 10])
                     .build())

        # Create cross-validator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1")

        cv = CrossValidator(estimator=self.pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3,
                            parallelism=2)

        return cv

    def train_model(self, pipeline, training_data):
        """
        Train the model using the provided pipeline and data.

        Args:
            pipeline: ML Pipeline or CrossValidator
            training_data: Training DataFrame

        Returns:
            Trained model
        """
        try:
            logger.info("Training model...")
            self.model = pipeline.fit(training_data)
            logger.info("Model training complete")

            # If using CrossValidator, get the best model
            if hasattr(self.model, 'bestModel'):
                self.model = self.model.bestModel

            return self.model
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def evaluate_model(self, model, test_data):
        """
        Evaluate the model and compute various metrics.

        Args:
            model: Trained model
            test_data: Test DataFrame

        Returns:
            Dictionary of evaluation metrics
        """
        if model is None or test_data is None:
            return None

        try:
            # Make predictions
            predictions = model.transform(test_data)

            # Function to calculate metrics with different metricNames
            def calculate_metric(evaluator, metric_name):
                evaluator.setMetricName(metric_name)
                return evaluator.evaluate(predictions)

            # Create evaluator
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction")

            # Calculate metrics
            metrics = {
                'accuracy': calculate_metric(evaluator, 'accuracy'),
                'precision': calculate_metric(evaluator, 'weightedPrecision'),
                'recall': calculate_metric(evaluator, 'weightedRecall'),
                'f1': calculate_metric(evaluator, 'f1')
            }

            # Log metrics
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name.capitalize()}: {value:.4f}")

            # Convert to Pandas for visualization (optional)
            # self._visualize_results(predictions)

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None

    def _visualize_results(self, predictions):
        """
        Create visualizations of model results.

        Args:
            predictions: DataFrame with predictions
        """
        try:
            # Convert to pandas for visualization
            pdf = predictions.select("genre", "prediction", "label").toPandas()

            # Count distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(data=pdf, x="genre")
            plt.title("Genre Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("genre_distribution.png")

            # Confusion matrix could be added here
        except Exception as e:
            logger.warning(f"Visualization error: {str(e)}")

    def save_model(self, model, output_path):
        """
        Save the model to disk.

        Args:
            model: Model to save
            output_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if model is None:
            return False

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save model
            model.write().overwrite().save(output_path)
            logger.info(f"Model saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, input_path):
        """
        Load a model from disk.

        Args:
            input_path: Path to the saved model

        Returns:
            Loaded model or None if error
        """
        from pyspark.ml import PipelineModel

        try:
            self.model = PipelineModel.load(input_path)
            logger.info(f"Model loaded from {input_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def predict_genre(self, text, release_year=None):
        """
        Predict the genre of a song based on its lyrics.

        Args:
            text: Song lyrics
            release_year: Release year (optional)

        Returns:
            Predicted genre
        """
        if self.model is None:
            logger.error("No model loaded. Please train or load a model first.")
            return None

        try:
            # Create a DataFrame with the input
            data = [(None, None, release_year or 2000, None, text)]
            columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
            df = self.spark.createDataFrame(data, columns)

            # Make prediction
            prediction = self.model.transform(df)

            # Get the predicted label index
            pred_index = prediction.select("prediction").collect()[0][0]

            # Convert back to genre name
            # This requires knowing the mapping from index to label
            # A more robust solution would use IndexToString transformer

            return f"Predicted genre index: {pred_index}"
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None

    def cleanup(self):
        """Stop the SparkSession when done."""
        if self.spark:
            self.spark.stop()
            logger.info("SparkSession stopped")


def main():
    """Main execution function."""
    # Create the classifier
    classifier = MusicGenreClassifier()

    # Define data paths
    mendeley_path = "/Users/shalukakalpajith/PycharmProjects/dataSetCreation/data/Mendeley_dataset.csv"
    student_path = "/Users/shalukakalpajith/PycharmProjects/dataSetCreation/data/Student_dataset.csv"
    required_columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]

    try:
        # Load and prepare Mendeley data
        logger.info("Loading Mendeley dataset...")
        mendeley_df = classifier.load_data(mendeley_path)
        mendeley_df = classifier.prepare_data(mendeley_df, required_columns)

        if mendeley_df is None:
            logger.error("Failed to load or prepare Mendeley dataset. Exiting.")
            return

        # Split for training
        training_data, test_data = mendeley_df.randomSplit([0.8, 0.2], seed=12345)
        logger.info(f"Training data: {training_data.count()} records")
        logger.info(f"Test data: {test_data.count()} records")

        # Create and train initial model
        pipeline = classifier.create_pipeline(training_data)
        model = classifier.train_model(pipeline, training_data)

        # Evaluate initial model
        if model:
            metrics = classifier.evaluate_model(model, test_data)
            classifier.save_model(model, "models/initial_model")

        # Load and prepare Student data
        logger.info("Loading Student dataset...")
        student_df = classifier.load_data(student_path)
        student_df = classifier.prepare_data(student_df, required_columns)

        if student_df is None:
            logger.warning("Failed to load or prepare Student dataset. Only using Mendeley data.")
        else:
            # Merge datasets
            merged_data = mendeley_df.union(student_df)
            logger.info(f"Merged data: {merged_data.count()} records")

            # Create and train merged model
            merged_pipeline = classifier.create_pipeline(merged_data, hyperparameter_tuning=True)
            merged_model = classifier.train_model(merged_pipeline, merged_data)

            if merged_model:
                # No test data for merged model (using all data for training)
                classifier.save_model(merged_model, "models/merged_model")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Clean up resources
        classifier.cleanup()


if __name__ == "__main__":
    main()