from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import pickle

app = FastAPI()

# Enable CORS so your HTML frontend can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Encoder
with open('isl_model.p', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    label_encoder = saved_data['encoder']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

def extract_hand_features(hand_landmarks):
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    coords = []
    pts = []
    
    for lm in hand_landmarks.landmark:
        nx, ny, nz = lm.x - base_x, lm.y - base_y, lm.z - base_z
        coords.extend([nx, ny, nz])
        pts.append(np.array([nx, ny, nz]))
        
    fingertip_indices = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(pts[idx]) for idx in fingertip_indices]
    
    inter_finger = []
    for i in range(len(fingertip_indices)):
        for j in range(i + 1, len(fingertip_indices)):
            inter_finger.append(np.linalg.norm(pts[fingertip_indices[i]] - pts[fingertip_indices[j]]))
            
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2)
    norm_val = np.linalg.norm(normal)
    normal = list(normal / norm_val) if norm_val > 0 else [0.0, 0.0, 0.0]
            
    return coords + distances + inter_finger + normal

@app.post("/predict")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Mirror flip to match local behavior
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    response_data = {"prediction": "No Hands Detected", "confidence": 0.0, "status": "no_hands"}
    
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        
        landmarks = []
        for hand in sorted_hands:
            landmarks.extend(extract_hand_features(hand))
        
        expected_len = 162
        if len(landmarks) < expected_len:
            landmarks.extend([0.0] * (expected_len - len(landmarks)))
        elif len(landmarks) > expected_len:
            landmarks = landmarks[:expected_len]
            
        probs = model.predict_proba([landmarks])[0]
        max_prob = np.max(probs)
        pred_class_id = np.argmax(probs)
        prediction = label_encoder.inverse_transform([pred_class_id])[0]
        
        if max_prob > 0.25:
            response_data = {
                "prediction": str(prediction),
                "confidence": float(max_prob * 100),
                "status": "success"
            }
        else:
            response_data = {
                "prediction": "Analyzing Sign...",
                "confidence": float(max_prob * 100),
                "status": "analyzing"
            }
            
    return response_data