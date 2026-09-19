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


# Load model and data flexibly
@st.cache_resource
def load_model_and_data():
  with open("isl_model.p", "rb") as f:
    data = pickle.load(f)
  return data


try:
  raw_data = load_model_and_data()
except Exception as e:
  st.error(f"Error loading model file (`isl_model.p`): {e}")
  st.stop()

# Parse model and labels dynamically
model = None
labels = None

if isinstance(raw_data, dict):
  model = raw_data.get("model") or raw_data.get("clf")
  labels = (
      raw_data.get("labels_encoder")
      or raw_data.get("label_encoder")
      or raw_data.get("labels")
      or raw_data.get("output_labels")
  )
else:
  model = raw_data

# Fallback: if separate labels weren't found, try getting classes directly from the model
if labels is None and model is not None and hasattr(model, "classes_"):
  labels = model.classes_

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)

# Streamlit camera input widget
img_file_buffer = st.camera_input("Take a picture of your sign")

if img_file_buffer is not None:
  # Convert uploaded buffer to an OpenCV image
  bytes_data = img_file_buffer.getvalue()
  cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

  img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  results = hands.process(img_rgb)
  data_aux = []

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      # Draw hand landmarks for visual feedback
      mp_drawing.draw_landmarks(
          cv2_img,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing.DrawingSpec(
              color=(0, 255, 0), thickness=2, circle_radius=2
          ),
          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
      )

      x_, y_ = [], []
      for i in range(len(hand_landmarks.landmark)):
        x_.append(hand_landmarks.landmark[i].x)
        y_.append(hand_landmarks.landmark[i].y)

      # Extract relative coordinates (relative to wrist landmark 0)
      for i in range(len(hand_landmarks.landmark)):
        data_aux.append(hand_landmarks.landmark[i].x - x_[0])
        data_aux.append(hand_landmarks.landmark[i].y - y_[0])

    # Display processed image with drawn landmarks
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
      pred_idx = prediction[0]

      # Decode label safely
      predicted_name = str(pred_idx)
      if labels is not None:
        if hasattr(labels, "inverse_transform"):
          predicted_name = labels.inverse_transform(prediction)[0]
        elif isinstance(labels, (list, np.ndarray)) and int(pred_idx) < len(
            labels
        ):
          predicted_name = labels[int(pred_idx)]
        elif isinstance(labels, dict):
          predicted_name = labels.get(pred_idx, str(pred_idx))

      st.success(f"### Prediction: {predicted_name}")

    except Exception as e:
      st.error(f"Prediction Error: {e}")
  else:
    st.warning(
        "No hands detected in the frame. Please make sure your hand is clearly"
        " visible."
    )