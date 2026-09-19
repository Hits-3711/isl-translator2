import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="ISL Translator", page_icon="🤟", layout="centered"
)

st.title("🤟 Indian Sign Language (ISL) Translator")
st.write(
    "Capture an image of your sign using the camera below to translate it"
    " instantly!"
)


# Load model and label encoder safely
@st.cache_resource
def load_model():
  with open("isl_model.p", "rb") as f:
    data = pickle.load(f)
  return data["model"], data["labels_encoder"]


try:
  model, label_encoder = load_model()
except Exception as e:
  st.error(f"Error loading model file (`isl_model.p`): {e}")
  st.stop()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)

# Streamlit camera input widget
img_file_buffer = st.camera_input("Take a picture of your sign")

if img_file_buffer is not None:
  # Convert the uploaded buffer to an OpenCV image
  bytes_data = img_file_buffer.getvalue()
  cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

  img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  results = hands.process(img_rgb)
  data_aux = []

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      # Draw hand landmarks on the image for visual feedback
      mp_drawing.draw_landmarks(
          cv2_img,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing.DrawingSpec(
              color=(0, 255, 0), thickness=2, circle_radius=2
          ),
          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
      )

      # Extract X and Y coordinates for all 21 landmarks per hand
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x)
        data_aux.append(y)

    # Display the processed image with drawn landmarks
    st.image(
        cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
    )

    # Ensure feature vector matches the model's expected size (162 features)
    EXPECTED_FEATURES = 162
    if len(data_aux) < EXPECTED_FEATURES:
      data_aux.extend([0.0] * (EXPECTED_FEATURES - len(data_aux)))
    elif len(data_aux) > EXPECTED_FEATURES:
      data_aux = data_aux[:EXPECTED_FEATURES]

    try:
      # Make prediction
      prediction = model.predict([np.asarray(data_aux)])

      # Decode prediction if label encoder is available
      if label_encoder is not None:
        predicted_character = label_encoder.inverse_transform(prediction)[0]
      else:
        predicted_character = str(prediction[0])

      st.success(f"### Prediction: {predicted_character}")

    except Exception as e:
      st.error(f"Prediction Error: {e}")
  else:
    st.warning(
        "No hands detected in the frame. Please make sure your hand is clearly"
        " visible."
    )