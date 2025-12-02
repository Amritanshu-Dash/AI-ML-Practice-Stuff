# Press 1 = Scan photos in "recognising" folder
# Press 2 = Live webcam

import insightface
from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np

# Folders
known_folder = "known_people"      # Put naruto.jpg, elon_musk.jpg here
test_folder = "recognising"        # Put test photos here
result_folder = "results"

os.makedirs(known_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Load InsightFace model (super fast, runs on CPU or GPU)
print("Loading InsightFace model... (takes 3 seconds)")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces
known_embeddings = []
known_names = []

print("\nLoading known faces...")
for file in os.listdir(known_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_folder, file)
        img = cv2.imread(path)
        faces = app.get(img)
        if len(faces) > 0:
            known_embeddings.append(faces[0].embedding)
            name = os.path.splitext(file)[0]
            known_names.append(name)
            print(f"Loaded → {name}")
        else:
            print(f"No face in {file}")

if len(known_names) == 0:
    print("Add photos to 'known_people' folder!")
    exit()

print(f"\nReady! Known: {known_names}\n")

# Function to find match
def find_match(embedding):
    distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
    min_dist = min(distances)
    if min_dist < 0.5:  # Threshold (lower = stricter)
        return known_names[distances.index(min_dist)], min_dist
    return "Unknown", min_dist

# Main menu
while True:
    print("═" * 55)
    print("    INSIGHTFACE NARUTO & ELON DETECTOR (2025)")
    print("═" * 55)
    print("1 → Scan photos in 'recognising' folder")
    print("2 → Live webcam mode")
    print("Q → Quit")
    print("═" * 55)
    choice = input("Choose (1/2/Q): ").strip().lower()

    if choice == '1':
        photos = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not photos:
            print("No photos in 'recognising' folder!")
            input("Press Enter...")
            continue

        for photo in photos:
            path = os.path.join(test_folder, photo)
            print(f"\nScanning → {photo}")
            img = cv2.imread(path)
            faces = app.get(img)

            for face in faces:
                name, dist = find_match(face.embedding)
                print(f"Detected: {name} (confidence: {1-dist:.3f})")

                # Draw box
                bbox = face.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                cv2.putText(img, f"{name}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

            # Save result
            out_path = os.path.join(result_folder, "RESULT_" + photo)
            cv2.imwrite(out_path, img)
            print(f"Saved → {out_path}")

        print("\nAll done! Check 'results' folder")
        input("Press Enter to continue...")

    elif choice == '2':
        print("\nStarting webcam... Press 'Q' to quit")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break

            faces = app.get(frame)
            for face in faces:
                name, dist = find_match(face.embedding)
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                cv2.putText(frame, name, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("InsightFace Live - Press Q", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == 'q':
        print("Rasengan complete. Bye bro!")
        break
