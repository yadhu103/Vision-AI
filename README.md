# Vision-AI
This project is a simple yet powerful face recognition web application that uses a pre-trained MobileNetV2 model to classify images captured from a webcam. The app is built using Flask for the backend, OpenCV for image processing, and TensorFlow/Keras for the machine learning model.

Features
Real-Time Face Recognition: Captures images from the user's webcam and classifies them into one of the predefined categories.
Pre-Trained Model: Utilizes a MobileNetV2 model fine-tuned on a custom dataset to recognize specific individuals.
Web Interface: Provides an easy-to-use web interface for capturing images and displaying predictions.
Cross-Origin Support: Enabled CORS to allow the web app to communicate with the Flask backend seamlessly.
Project Structure
app.py: The Flask backend that handles image processing and prediction.
face_recog.html: The frontend HTML file that includes the webcam interface and handles user interactions.
person_classifier_model.h5: The pre-trained Keras model used for predictions.
static/: Directory containing any static assets like CSS or JavaScript files.
templates/: Directory containing HTML templates.
How It Works
The user accesses the web app and enables the webcam.
Upon clicking the "Search" button, a snapshot is taken and sent to the Flask backend.
The image is processed and fed into the machine learning model to predict the class.
The predicted class is returned and displayed on the web page.
Getting Started
Clone the repository.
Install the required dependencies listed in requirements.txt.
Run the Flask app using python app.py.
Open a browser and go to http://127.0.0.1:5000 to access the app.
Future Improvements
Model Optimization: Improve the accuracy and speed of the model.
UI Enhancements: Create a more user-friendly and visually appealing interface.
Additional Features: Add the ability to train the model with new images directly from the app.
