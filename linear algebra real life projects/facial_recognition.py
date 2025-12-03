# ===================================================================
# NARUTO & ELON MUSK FACE DETECTOR — FULLY EXPLAINED VERSION (2025), well this is specific you can give your own images and call it whatever you want
# This code uses InsightFace — the strongest face recognition in the world
# You put ALL Naruto photos in: known_people/naruto/
# You put ALL Elon photos in:   known_people/elon_musk/
# It will train on 100 photos if you want — and still show only "naruto"
# ===================================================================

import insightface                     # Main AI model library (like brain of the system)
from insightface.app import FaceAnalysis  # The actual face detector + encoder
import cv2                             # OpenCV — for reading images and showing webcam
import os                              # Operating System — to work with folders and files
import numpy as np                     # Math library — for calculating face similarity

# ===================================================================
# STEP 1: DEFINE FOLDERS (where everything is stored)
# ===================================================================
known_folder = "known_folder_or_training_folder"  # MAIN folder: contains subfolders like naruto/, elon_musk/
test_folder = "test_folder_or_recognising "        # Folder where you put photos you want to TEST
result_folder = "results"              # Folder where output images (with green boxes) will be saved

# Create these folders automatically if they don't exist yet
os.makedirs(known_folder, exist_ok=True)    # exist_ok=True means "don't crash if already exists"
os.makedirs(test_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

print("Folders ready: known_people, recognising, results")

# ===================================================================
# STEP 2: LOAD THE INSIGHTFACE AI MODEL (this is the magic brain)
# ===================================================================
print("\nLoading the super-smart face recognition model... (takes ~3 seconds)")
app = FaceAnalysis(
    name="buffalo_l",                  # Best free model in 2025 — detects faces perfectly
    providers=['CPUExecutionProvider'] # Use CPU (works on all Macs). If you have M1/M2, it uses GPU automatically!
)
app.prepare(ctx_id=0, det_size=(640, 640))  # Prepare the model with good detection size

print("AI Model loaded successfully!")

# ===================================================================
# STEP 3: TRAIN THE SYSTEM ON ALL YOUR KNOWN PEOPLE
# We will store face "DNA" (called embeddings) in a dictionary
# Dictionary key = person name (like "naruto")
# Value = list of all face embeddings from that person
# ===================================================================
known_embeddings = {}  # This will become: {"naruto": [emb1, emb2, ...], "elon_musk": [...]}

print("\n" + "="*60)
print("TRAINING PHASE: Reading all photos from known_people folder")
print("="*60)

# Go through every item inside the known_people folder
for item in os.listdir(known_folder):
    full_path = os.path.join(known_folder, item)  # Full path to file/folder

    # CASE 1: It's a FOLDER (like naruto/ or elon_musk/)
    if os.path.isdir(full_path):
        person_name = item.lower().strip()                # Folder name becomes the person's name
        print(f"\nFound a person folder: {person_name.upper()}")

        # Make sure this person exists in our dictionary
        if person_name not in known_embeddings:
            known_embeddings[person_name] = []

        # Now loop through every image inside this person's folder
        for filename in os.listdir(full_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(full_path, filename)
                image = cv2.imread(image_path)     # Read the image

                faces = app.get(image)             # Let AI find faces in this photo

                if len(faces) > 0:                 # If at least one face found
                    embedding = faces[0].embedding # Get the face "DNA" (512 numbers)
                    known_embeddings[person_name].append(embedding)
                    print(f"  Trained photo: {filename} → labeled as '{person_name}'")
                else:
                    print(f"  No face found in: {filename} (skipped)")

    # CASE 2: It's a single file directly in known_people (like naruto.jpg)
    elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(full_path)
        faces = app.get(image)
        if len(faces) > 0:
            name = os.path.splitext(item)[0].lower()  # Remove .jpg and use filename as name
            if name not in known_embeddings:
                known_embeddings[name] = []
            known_embeddings[name].append(faces[0].embedding)
            print(f"  Trained single file: {item} → labeled as '{name}'")

# Final check: did we train anything?
if not known_embeddings:
    print("ERROR: No faces were trained!")
    print("Put photos in known_people/naruto/ or known_people/elon_musk/")
    input("Press Enter to exit...")
    exit()

# Show training summary
print("\n" + "="*60)
print("TRAINING COMPLETE! Here are the people I know:")
print("="*60)
for name, embeddings_list in known_embeddings.items():
    print(f"  → {name.upper():15} : {len(embeddings_list)} photos trained")
print("="*60)

# ===================================================================
# STEP 4: FUNCTION TO COMPARE A NEW FACE WITH ALL KNOWN FACES
# This decides: "Is this face Naruto? Elon? Or Unknown?"
# ===================================================================
def find_best_match(new_embedding):
    best_name = "Unknown"
    best_distance = 99.0                    # Higher = worse match

    # Compare the new face with EVERY known embedding
    for person_name, embedding_list in known_embeddings.items():
        for known_embedding in embedding_list:
            # Calculate how similar two faces are (0 = identical, 1+ = different)
            distance = np.linalg.norm(new_embedding - known_embedding)

            if distance < best_distance:    # Found a better match!
                best_distance = distance
                best_name = person_name

    # Only accept if similarity is good enough (threshold = 0.55)
    if best_distance < 0.6:
        return best_name, 1 - best_distance   # Return name + confidence (0-1) so best_distance is lower is better and it confidence is higher is better 1 - distance
    else:
        return "Unknown", best_distance

# ===================================================================
# STEP 5: MAIN MENU — CHOOSE WHAT TO DO
# ===================================================================
while True:
    print("\n" + "█" * 70)
    print("       NARUTO & ELON MUSK DETECTOR — FULLY TRAINED & READY")
    print("█" * 70)
    print("1 → Scan all photos in 'recognising' folder")
    print("2 → Start live webcam detection")
    print("Q → Quit the program")
    print("█" * 70)
    choice = input("\nWhat do you want to do? (1/2/Q): ").strip().lower()

    # ===================================================================
    # OPTION 1: SCAN PHOTOS IN recognising/ FOLDER
    # ===================================================================
    if choice == '1':
        print("\nScanning all test photos...")
        test_photos = [f for f in os.listdir(test_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not test_photos:
            print("No photos found in 'recognising' folder!")
            print("Put some photos there first!")
            input("Press Enter to go back...")
            continue

        for photo_name in test_photos:
            photo_path = os.path.join(test_folder, photo_name)
            print(f"\nAnalyzing: {photo_name}")
            image = cv2.imread(photo_path)
            detected_faces = app.get(image) # Find all faces in this photo
            print(f"  Found {len(detected_faces)} face(s) in {photo_name}")
            if len(detected_faces) == 0:
                print("  NO FACE DETECTED — check photo quality!")
                # Still save image so you can see
                cv2.putText(image, "NO FACE FOUND", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            else:
                for face in detected_faces:
                    name, confidence = find_best_match(face.embedding)
                    detected_name = name.strip().lower()
                    display_name = name
                    print(f"  DETECTED → photo : {photo_name} : {display_name} (confidence: {confidence:.3f})")
                    # Draw green box around face
                    bbox = face.bbox.astype(int)
                    is_known = any(detected_name == known.strip().lower() for known in known_embeddings.keys())
                    if is_known:
                        color = (0, 255, 0)  # Green for known faces
                        label = f"{display_name} ({confidence:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown faces
                        label = "Unknown"

                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=6)
                    cv2.putText(image, label, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

            # Save result
            output_path = os.path.join(result_folder, "RESULT_" + photo_name)
            cv2.imwrite(output_path, image)
            print(f"  Result saved → {output_path}")

        print("\nAll photos scanned! Check the 'results' folder")
        input("Press Enter to continue...")

    # ===================================================================
    # OPTION 2: LIVE WEBCAM MODE
    # ===================================================================
    elif choice == '2':
        print("\nStarting webcam... Look at the camera!")
        print("Press 'Q' on your keyboard to stop")

        # Open your MacBook's camera (0 = default camera)
        cap = cv2.VideoCapture(0)

        # Optional: make window smaller (easier to see)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

        while True:
            ret, frame = cap.read()        # Read one frame from camera
            if not ret:                    # If camera fails
                print("Cannot access webcam!")
                break

            faces = app.get(frame)         # Find all faces in current frame

            for face in faces:
                name, confidence = find_best_match(face.embedding)

                # Draw box and name
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = name
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)

            # Show the video feed
            cv2.imshow("LIVE FACE RECOGNITION — Press Q to quit", frame)

            # Stop if user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")

    # ===================================================================
    # OPTION Q: QUIT
    # ===================================================================
    elif choice == 'q':
        print("\nThank you for using the Face Detector!")
        break

    else:
        print("Invalid choice! Please type 1, 2, or Q")
