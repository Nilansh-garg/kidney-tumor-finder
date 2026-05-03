import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
import os

MODEL_PATH = "artifacts/training/model.keras"

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        hf_hub_download(
            repo_id="Nilansh13/kidney-tumor-model",
            filename="model.keras",
            local_dir="artifacts/training",
            repo_type="model"
        )
    return load_model(MODEL_PATH)

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
