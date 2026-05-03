from unittest import result
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import keras

# Custom loader that ignores quantization_config
def load_model_ignore_quantization(path):
    return load_model(path, custom_objects={}, compile=False)

class predict_pipeline:
    def __init__(self, filename):
        self.filename = filename
              
    def predict(self):
        try:
            # Try with compile=False first
            model = load_model(
                os.path.join("artifacts","training","model.h5"),
                compile=False
            )
        except Exception as e:
            print(f"First attempt failed: {e}")
            # If that fails, try loading weights separately
            try:
                from tensorflow.keras.applications import VGG16
                base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                model = load_model(os.path.join("artifacts","training","model.h5"))
            except:
                raise
        
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image,axis = 0)
        raw_prediction = model.predict(test_image)
        print(raw_prediction)
        result = (raw_prediction[0][0] >= 0.5).astype(int)
         
        print(result)
                 
        prob = float(raw_prediction[0][0])
        confidence = prob if result == 1 else 1 - prob
                 
        if result == 1:
            prediction = "Tumor"
            return [{"class": prediction, "confidence": confidence}]
        else:
            prediction = "Normal"
            return [{"class": prediction, "confidence": confidence}]
