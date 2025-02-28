from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import face_recognition
import numpy as np
import cv2
import os, math


app = FastAPI()

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lists for known face encodings and names
known_face_encodings = []
known_face_names = []


# Load and encode known faces from the 'faces' folder
def encode_faces():
    for img_file in os.listdir('faces'):
        image = face_recognition.load_image_file(f'faces/{img_file}')
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            # Remove file extension for clarity
            known_face_names.append(os.path.splitext(img_file)[0])
    print("Known faces:", known_face_names)


encode_faces()


@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    # Convert image data to a numpy array and decode it
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    faces = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                threshold = 0.6
                distance = face_distances[best_match_index]

                # Compute confidence based on face distance
                range_val = (1.0 - threshold)
                linear_val = (1.0 - distance) / (range_val * 2.0)
                if distance > threshold:
                    confidence = round(linear_val * 100, 2)
                else:
                    value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
                    confidence = round(value, 2)

        # Scale back the face locations to original frame size
        face_location = {
            "top": top * 4,
            "right": right * 4,
            "bottom": bottom * 4,
            "left": left * 4
        }
        faces.append({"name": name, "confidence": confidence, "location": face_location})
    print(faces)
    return {"faces": faces}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
