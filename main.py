import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import csv

# Create a folder to save images if it doesn't exist
folder_name = "captured_images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Path to the CSV file
csv_file = os.path.join(folder_name, "camera_record.csv")

video_capture = cv2.VideoCapture(0)

path = 'C:\\Users\\shruti\\OneDrive\\Desktop\\shru\\images'

# Load known faces and their names
images = []
known_face_names = []

myImages = os.listdir(path)
for image in myImages:
    currentImage = cv2.imread(f'{path}/{image}')
    images.append(currentImage)
    known_face_names.append(image.split('.')[0])

def load_known_faces(images):
    encodingList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(image)[0]
        encodingList.append(encoding)

    return encodingList

def save_entry(name):
    with open(csv_file, 'a+', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        date_now =now.strftime('%Y-%m-%d' )
        dtString =now.strftime('%H:%M:%S')
        writer.writerow([name, date_now, dtString])

# before storing data check if it exists or not
def entry_exists(name):
    if not os.path.exists(csv_file):
        return False
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                return True
    return False

encodeListKnown=load_known_faces(images)
print("Encoding completed...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        # See if the face is a match for any known face(s)
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        print(matches)
        name="unknown"
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)
        
        # If a match was found in known_face_encodings use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Check if the entry already exists to avoid duplicates
        if name != "Unknown" and not entry_exists(name):
            # Save the image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = os.path.join(folder_name, f"{name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            # Save the entry in the CSV file
            save_entry(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
