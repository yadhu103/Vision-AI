from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": "*"}})  

model = load_model("person_classifier_model.h5")

def preprocess_image(image):
    try:
        image_data = base64.b64decode(image)
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Failed to decode image.")
            return None

        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('face_recog.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    image_data = data['image']
    image = preprocess_image(image_data)
    
    if image is None:
        return jsonify({'error': 'Image preprocessing failed'}), 400
    
    prediction = model.predict(image)
    print("Raw prediction:", prediction)
    
    label = np.argmax(prediction, axis=1)[0]
    
    class_names = ['Father','Mother', 'Nandu', 'Yadhu']
    
    prediction_name = class_names[label] if 0 <= label < len(class_names) else "Unknown"
    print("Predicted class:", prediction_name)
    
    return jsonify({'prediction': prediction_name})

if __name__ == '__main__':
    app.run(debug=True) 
