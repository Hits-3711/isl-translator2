import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# 1. Page Configuration
st.set_page_config(page_title="ISL Live Translator", page_icon="🤟", layout="centered")

st.title("🤟 Indian Sign Language Live Translator")
st.write("Perform signs in front of your camera to translate them in real-time.")

# 2. Load the Model & Encoder Safely
@st.cache_resource
def load_model():
    model_path = "isl_model.p"  # Make sure isl_model.p is in the same folder as app.py
    if not os.path.exists(model_path):
        return None, None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle dictionary style or tuple style pickles depending on how it was saved
    if isinstance(data, dict):
        return data.get("model"), data.get("label_encoder")
    else:
        return data[0], data[1]

model, label_encoder = load_model()

if model is None:
    st.error("⚠️ `isl_model.p` not found! Please place your model file in the project folder.")
else:
    st.success("Model loaded successfully!")

# 3. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 4. WebRTC or Camera Input Setup
image_file = st.camera_input("Take a picture / Stream Camera")

if image_file is not None:
    # Convert the file buffer to an OpenCV image
    bytes_data = image_file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Flip frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    prediction_text = "No hand detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract coordinates matching feature extraction logic
            data_aux = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
                
            # Predict using the loaded model if feature length matches
            if model is not None and len(data_aux) > 0:
                try:
                    # Ensure input is shaped correctly as a 2D array for XGBoost
                    input_data = np.asarray(data_aux).reshape(1, -1)
                    
                    prediction = model.predict(input_data)
                    
                    # Handle Label Encoder or direct prediction output
                    if label_encoder is not None:
                        predicted_character = label_encoder.inverse_transform(prediction)
                        prediction_text = str(predicted_character[0])
                    else:
                        prediction_text = str(prediction[0])
                        
                except Exception as e:
                    prediction_text = f"Error: {str(e)}"

    # Display results in Streamlit UI
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    st.markdown(f"### Prediction: **{prediction_text}**")