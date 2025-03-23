# web_app.py
import os
import logging
from flask import Flask, request, render_template, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
import findspark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("webapp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MusicGenreWebApp:
    """Flask web application for music genre classification."""

    def __init__(self, model_path, host='0.0.0.0', port=8080):
        """Initialize the web application.

        Args:
            model_path: Path to the saved model
            host: Host to run the Flask app on
            port: Port to run the Flask app on
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.spark = None
        self.model = None
        self.labels = None

        # Initialize Spark and load model
        self._initialize_spark()
        success = self._load_model()

        if not success:
            logger.error("Failed to load model. Application cannot start.")
            raise RuntimeError("Model loading failed")

        # Setup routes
        self._setup_routes()

    def _initialize_spark(self):
        """Initialize Spark session with appropriate configuration."""
        try:
            # Initialize findspark to locate Spark
            findspark.init()

            # Create Spark session with memory and driver configurations
            self.spark = (SparkSession.builder
                          .appName("MusicGenreWebApp")
                          .config("spark.driver.memory", "4g")
                          .config("spark.executor.memory", "4g")
                          .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
                          .config("spark.ui.enabled", "false")  # Disable UI for web app context
                          .getOrCreate())

            logger.info("Spark session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spark: {str(e)}")
            raise

    def _load_model(self):
        """Load the trained PipelineModel and extract label mapping.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Ensure model path exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                return False

            # Load the PipelineModel
            logger.info(f"Loading model from {self.model_path}")
            self.model = PipelineModel.load(self.model_path)

            # Extract StringIndexerModel to get label mapping
            # Find the StringIndexer stage in the pipeline model
            string_indexer = None
            for stage in self.model.stages:
                if isinstance(stage, StringIndexerModel):
                    string_indexer = stage
                    break

            if string_indexer is None:
                # Attempt to load specifically from the path if not found in pipeline
                indexer_path = os.path.join(self.model_path, "stages/3_StringIndexer_*")
                import glob
                indexer_paths = glob.glob(indexer_path)
                if indexer_paths:
                    string_indexer = StringIndexerModel.load(indexer_paths[0])
                else:
                    logger.error("StringIndexer model not found in pipeline or at expected path")
                    return False

            # Extract labels from the StringIndexerModel
            self.labels = string_indexer.labels
            logger.info(f"Loaded model with {len(self.labels)} genre classes: {self.labels}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def _setup_routes(self):
        """Set up Flask routes."""

        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            if request.method == 'POST':
                return self._handle_prediction_request(request)
            return render_template('index.html')

        @self.app.route('/predict', methods=['POST'])
        def predict_api():
            """API endpoint for predictions."""
            return self._handle_prediction_request(request)

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({"status": "healthy"})

    def _handle_prediction_request(self, request):
        """Process prediction requests.

        Args:
            request: Flask request object

        Returns:
            JSON response with prediction results
        """
        try:
            # Extract lyrics from form or JSON
            if request.is_json:
                data = request.get_json()
                lyrics = data.get('lyrics', '')
            else:
                lyrics = request.form.get('lyrics', '')

            # Validate input
            if not lyrics or len(lyrics) < 10:
                return jsonify({
                    "error": "Please provide lyrics with at least 10 characters"
                }), 400

            # Create a DataFrame for the input lyrics
            input_df = self.spark.createDataFrame([(lyrics,)], ["lyrics"])

            # Make predictions
            predictions = self.model.transform(input_df)

            # Extract prediction results
            if predictions.count() == 0:
                raise ValueError("Prediction failed - no results returned")

            # Get predicted label and probability
            prediction_row = predictions.select("prediction", "probability").collect()[0]
            predicted_label_index = int(prediction_row[0])
            probabilities = prediction_row[1].toArray().tolist()

            # Map predicted index to label
            if predicted_label_index < len(self.labels):
                predicted_label = self.labels[predicted_label_index]
            else:
                logger.error(f"Invalid label index: {predicted_label_index}")
                predicted_label = "Unknown"

            # Format probability values for readability
            formatted_probs = [round(prob * 100, 2) for prob in probabilities]

            # Prepare response data
            chart_data = {
                'labels': self.labels,
                'data': formatted_probs,
                'predictedLabel': predicted_label,
                'confidence': round(probabilities[predicted_label_index] * 100, 2)
            }

            logger.info(f"Prediction: {predicted_label} with {chart_data['confidence']}% confidence")
            return jsonify(chart_data)

        except Exception as e:
            logger.error(f"Error processing prediction request: {str(e)}")
            return jsonify({"error": "An error occurred processing your request"}), 500

    def run(self, debug=False):
        """Run the Flask web application.

        Args:
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting web application on {self.host}:{self.port}")
        self.app.run(debug=debug, host=self.host, port=self.port)

    def shutdown(self):
        """Shutdown the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def create_sample_template():
    """Create a sample index.html template if it doesn't exist."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)

    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Music Genre Classifier</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { padding: 20px; }
        #result-container { display: none; margin-top: 20px; }
        #prediction-result { font-size: 24px; font-weight: bold; margin-bottom: 15px; }
        .loader { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 2s linear infinite; display: none; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Music Genre Classifier</h1>

        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Enter Song Lyrics</h5>
                <form id="lyrics-form" method="post">
                    <div class="mb-3">
                        <textarea class="form-control" id="lyrics" name="lyrics" rows="8" placeholder="Paste song lyrics here..."></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Predict Genre</button>
                </form>
            </div>
        </div>

        <div class="loader" id="loader"></div>

        <div id="result-container" class="card">
            <div class="card-body">
                <h5 class="card-title">Prediction Results</h5>
                <div id="prediction-result"></div>
                <canvas id="genreChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('lyrics-form');
        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('result-container');
        const predictionResult = document.getElementById('prediction-result');
        let genreChart = null;

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            // Show loader, hide results
            loader.style.display = 'block';
            resultContainer.style.display = 'none';

            // Get form data
            const formData = new FormData(form);

            try {
                // Send request
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Server error');
                }

                // Process response
                const data = await response.json();

                // Display results
                predictionResult.textContent = `Predicted Genre: ${data.predictedLabel} (${data.confidence}% confidence)`;

                // Create chart
                createChart(data);

                // Show results, hide loader
                resultContainer.style.display = 'block';
                loader.style.display = 'none';

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during prediction');
                loader.style.display = 'none';
            }
        });

        function createChart(data) {
            const ctx = document.getElementById('genreChart').getContext('2d');

            // Destroy existing chart if any
            if (genreChart) {
                genreChart.destroy();
            }

            // Create new chart
            genreChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Genre Probability (%)',
                        data: data.data,
                        backgroundColor: data.labels.map(label => 
                            label === data.predictedLabel ? 'rgba(54, 162, 235, 0.8)' : 'rgba(201, 203, 207, 0.8)'
                        ),
                        borderColor: data.labels.map(label => 
                            label === data.predictedLabel ? 'rgb(54, 162, 235)' : 'rgb(201, 203, 207)'
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Probability (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Genre'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Genre Prediction Probabilities'
                        },
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
            """)
        logger.info(f"Created sample template at {template_path}")


if __name__ == '__main__':
    # Define path to model
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/merged_model')
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8080))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Create sample template if needed
    create_sample_template()

    try:
        # Create and run the web app
        app = MusicGenreWebApp(MODEL_PATH, host=HOST, port=PORT)
        app.run(debug=DEBUG)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        exit(1)
    finally:
        # This won't be reached in normal operation due to Flask's blocking run()
        logger.info("Application shutting down")