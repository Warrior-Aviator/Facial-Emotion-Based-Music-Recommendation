# 🎭 Facial Emotion–Based Music Recommendation System
A real-time system that captures the user's facial emotions via webcam using MediaPipe, classifies them using a trained deep learning model, and recommends music (via YouTube) accordingly using Streamlit UI. The system supports multiple emotion categories and allows filtering music recommendations by language and singer.

# 📌 Features
🔴 Real-time facial emotion detection using webcam.

🧠 Emotion classification using a custom-trained deep learning model (model.h5).

🎶 Music recommendation through YouTube based on detected emotion.

🌐 Streamlit-powered interactive interface for smooth user experience.

💾 Emotion data collection and training pipeline included for easy customization.

# 🧠 Emotion Classes
The model supports the following emotions:

Happy

Sad

Angry

Surprise

Neutral

# 📁 Repository Structure
text
Copy
Edit
├── data_collection.py       # Collects facial landmarks data for training
├── data_training.py         # Trains a neural network model from collected .npy files
├── inference.py             # Real-time emotion prediction via webcam (CLI interface)
├── Music.py                 # Streamlit app for webcam-based emotion recognition + YouTube recommendations
├── Emotion.npy              # Temporary file to store predicted emotion
├── model.h5                 # Trained Keras model
├── labels.npy               # List of emotion labels
├── *.npy                    # Collected emotion data (Happy.npy, Sad.npy, etc.)
├── Music.txt                # Optional: music reference text
├── emotion.jpg              # Reference image (not used in code)
├── README.md                # This file
🛠️ Requirements
# Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Your requirements.txt should include:

txt
Copy
Edit
opencv-python
numpy
mediapipe
tensorflow
keras
streamlit
streamlit-webrtc
🚀 How to Run
# 1. Run the Streamlit App
For a full UI with live emotion recognition and music suggestion via YouTube:

bash
Copy
Edit
streamlit run Music.py
Webcam starts

Enter language and singer (optional)

Click "Recommend Songs" to open music on YouTube

# 2. CLI-Based Emotion Detection (No UI)
bash
Copy
Edit
python inference.py
Webcam opens

Detected emotion is shown in the window

Closes when you press Esc

# 3. Collect Training Data
bash
Copy
Edit
python data_collection.py
Enter emotion name (e.g., Happy)

Webcam captures facial and hand landmarks

100 samples are saved in a .npy file

# 4. Train the Model
bash
Copy
Edit
python data_training.py
Trains a dense neural network on collected .npy files

Saves trained model as model.h5

Saves label mapping as labels.npy

# 📈 Model Architecture
Input: 166 landmarks from face and hands (x and y deltas)

Hidden Layers: Dense(512) → ReLU → Dense(256) → ReLU

Output Layer: Softmax over emotion classes

Optimizer: RMSprop

Loss: Categorical Crossentropy

# 🔮 Future Improvements
Integrate Spotify/YouTube APIs to auto-play music

Use CNN-based emotion recognition directly from images

Add voice input or text-based emotion selection

Mobile/web deployment with Flask or Heroku


