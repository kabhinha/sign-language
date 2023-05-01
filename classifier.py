from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

def load_modell(modelPath, labelsPath):
    np.set_printoptions(suppress=True)

    model = load_model(modelPath, compile=False)

    class_names = open(labelsPath, "r").readlines()
    return model, class_names

def getPrediction(image, model, class_names):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    result = {"CLass":class_name[2:], "Confidence":str(np.round(confidence_score * 100))[:-2]}
    return result