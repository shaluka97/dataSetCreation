# web_app.py
from flask import Flask, request, render_template, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel  # Import StringIndexerModel

app = Flask(__name__)

# Load the trained model (ensure paths are correct)
spark = SparkSession.builder.appName("MusicGenreWebApp").getOrCreate()
try:
    model = PipelineModel.load("merged_model")  # Load the *merged* model
    string_indexer_model = StringIndexerModel.load("merged_model/stages/3_StringIndexer_b488d9bb589f") #load the string indexer model to get the labels
    labels = string_indexer_model.labels
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error appropriately (e.g., exit the application)
    exit(1)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lyrics = request.form['lyrics']

        # Create a DataFrame for the input lyrics
        input_df = spark.createDataFrame([(lyrics,)], ["lyrics"])

        # Make predictions
        predictions = model.transform(input_df)
        # Collect probabilities and convert to list of floats
        probabilities = predictions.select("probability").collect()[0][0].toArray().tolist()

        #get predicted label

        predicted_label_index = int(predictions.select("prediction").collect()[0][0])
        predicted_label = labels[predicted_label_index]

        # Prepare data for the chart
        chart_data = {
            'labels': labels,
            'data': probabilities,
            'predictedLabel': predicted_label  # Pass predicted label
        }

        return jsonify(chart_data)

    return render_template('index.html')  # Create index.html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Make it accessible on the network, use a specific port