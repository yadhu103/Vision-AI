## Vision-AI
This project is a simple yet powerful face recognition web application that uses a pre-trained MobileNetV2 model to classify images captured from a webcam. The app is built using Flask for the backend, OpenCV for image processing, and TensorFlow/Keras for the machine learning model.

**Features**: <br/>
-> Real-Time Face Recognition: Captures images from the user's webcam and classifies them into one of the predefined categories.<br/>
-> Pre-Trained Model: Utilizes a MobileNetV2 model fine-tuned on a custom dataset to recognize specific individuals.<br/>
-> Web Interface: Provides an easy-to-use web interface for capturing images and displaying predictions.<br/>
-> Cross-Origin Support: Enabled CORS to allow the web app to communicate with the Flask backend seamlessly.

**How It Works**: <br/>
-> The user accesses the web app and enables the webcam.<br/>
-> Upon clicking the "Search" button, a snapshot is taken and sent to the Flask backend.<br/>
-> The image is processed and fed into the machine learning model to predict the class.<br/>
-> The predicted class is returned and displayed on the web page.

**Future Improvements**: <br/>
-> Improve the accuracy of predictions<br/>
-> Make the UI more visually appealing and user-friendly<br/>
-> Expand the dataset.
