## Vision-AI
This project is a simple yet powerful face recognition web application that uses a pre-trained MobileNetV2 model to classify images captured from a webcam. The app is built using Flask for the backend, OpenCV for image processing, and TensorFlow/Keras for the machine learning model.#

#Features
-> Real-Time Face Recognition: Captures images from the user's webcam and classifies them into one of the predefined categories.
-> Pre-Trained Model: Utilizes a MobileNetV2 model fine-tuned on a custom dataset to recognize specific individuals.
-> Web Interface: Provides an easy-to-use web interface for capturing images and displaying predictions.
-> Cross-Origin Support: Enabled CORS to allow the web app to communicate with the Flask backend seamlessly.

#How It Works
-> The user accesses the web app and enables the webcam.
-> Upon clicking the "Search" button, a snapshot is taken and sent to the Flask backend.
-> The image is processed and fed into the machine learning model to predict the class.
-> The predicted class is returned and displayed on the web page.

#Future Improvements
-> Improve the accuracy of predictions
-> Make the UI more visually appealing and user-friendly
-> Expand the dataset.
