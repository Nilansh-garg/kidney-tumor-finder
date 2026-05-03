import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
import os

WEIGHTS_PATH = "artifacts/training/model_weights.weights.h5"
ARCH_PATH = "artifacts/training/model_architecture.json"

def get_model():
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs("artifacts/training", exist_ok=True)
        hf_hub_download(
            repo_id="Nilansh13/kidney-tumor-model",
            filename="model_weights.weights.h5",
            local_dir="artifacts/training",
            repo_type="model"
        )
        hf_hub_download(
            repo_id="Nilansh13/kidney-tumor-model",
            filename="model_architecture.json",
            local_dir="artifacts/training",
            repo_type="model"
        )
    with open(ARCH_PATH, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(WEIGHTS_PATH)
    return model

class predict_pipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = get_model()
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        raw_prediction = model.predict(test_image)
        result = (raw_prediction[0][0] >= 0.5).astype(int)
        prob = float(raw_prediction[0][0])
        confidence = prob if result == 1 else 1 - prob
        if result == 1:
            return [{"class": "Tumor", "confidence": confidence}]
        else:
            return [{"class": "Normal", "confidence": confidence}]
