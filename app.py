from flask import Flask, request, jsonify, render_template
import os
import traceback
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import predict_pipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = predict_pipeline(self.filename)

clApp = ClientApp()

@app.route("/", methods=["GET"])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route("/classify", methods=["GET"])
@cross_origin()
def classifyPage():
    return render_template("classify.html")

@app.route("/about", methods=["GET"])
@cross_origin()
def aboutPage():
    return render_template("about.html")

@app.route("/contact", methods=["GET"])
@cross_origin()
def contactPage():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        # Log full error details
        error_details = traceback.format_exc()
        print("PREDICTION ERROR:")
        print(error_details)
        return jsonify({
            "status": "error", 
            "message": str(e),
            "details": error_details
        }), 500

@app.route("/health", methods=["GET"])
@cross_origin()
def healthCheck():
    return jsonify({"status": "healthy"})

@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
