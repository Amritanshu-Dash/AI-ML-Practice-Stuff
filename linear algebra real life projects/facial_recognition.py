import face_recognition
import cv2
import numpy as np
import os

print("Looking for faces in known faces directory...")

# ------------------- Folders -------------------
known_people_folder = "known_people"      # Put Naruto, Elon photos here
recognising_folder = "recognising"        # Photos you want to test
results_folder = "results"                # Results will be saved here

# Create folders if they don't exist
os.makedirs(known_people_folder, exist_ok=True)
os.makedirs(recognising_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

known_faces_encodings = []
known_faces_names = []

# Check if known_people folder is empty
if len(os.listdir(known_people_folder)) == 0:
    print("Put photos in 'known_people' folder! Example: naruto.jpg, elon_musk.jpg")
    input("Press Enter to exit...")
    exit()

# Load known faces (FIXED: you wrote "known_faces" instead of "known_people")
for filename in os.listdir(known_people_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(known_people_folder, filename)
        print(f"Loading: {filename}")

        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if len(encoding) > 0:
            known_faces_encodings.append(encoding[0])
            name = os.path.splitext(filename)[0]
            known_faces_names.append(name)
            print(f"Successfully added → {name}")
        else:
            print(f"No face found in {filename}")

# If no faces loaded
if len(known_faces_encodings) == 0:
    print("No known faces found! Add clear photos to 'known_people' folder.")
    input("Press Enter to exit...")
    exit()
else:
    print(f"\nFound {len(known_faces_encodings)} known people: {known_faces_names}\n")

# ------------------- Main Menu Loop -------------------
while True:
    print("═" * 50)
    print("    NARUTO & ELON DETECTOR PRO")
    print("═" * 50)
    print("Press 1 → Check photos in 'recognising' folder")
    print("Press 2 → Live Webcam Mode")
    print("Press Q → Quit")
    print("═" * 50)

    choice = input("Choose (1/2/Q): ").strip().lower()

    # =============== OPTION 1: Check Photos ===============
    if choice == '1':
        print("\nScanning 'recognising' folder...")
        photos = [f for f in os.listdir(recognising_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(photos) == 0:
            print("No photos in 'recognising' folder!")
            input("Press Enter to go back...")
            continue

        for photo in photos:
            path = os.path.join(recognising_folder, photo)
            print(f"\nChecking → {photo}")

            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, locations)
            img = cv2.imread(path)

            if len(locations) == 0:
                print("No face found in this photo!")
                continue

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_faces_encodings, encoding, tolerance=0.5)
                name = "Unknown Person"

                distances = face_recognition.face_distance(known_faces_encodings, encoding)
                best_match = np.argmin(distances)

                if matches[best_match]:
                    name = known_faces_names[best_match]

                print(f"Detected: {name}")

                # Draw box and name (FIXED: cv2.rectangle takes points, not name)
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 4)
                cv2.rectangle(img, (left, bottom - 50), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (left + 10, bottom - 15),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)

            # Save result
            output_path = os.path.join(results_folder, "RESULT_" + photo)
            cv2.imwrite(output_path, img)
            print(f"Saved → {output_path}")

        print("\nAll photos processed! Check 'results' folder")
        input("Press Enter to continue...")

    # =============== OPTION 2: Live Webcam ===============
    elif choice == '2':
        print("\nStarting webcam... Press 'Q' to quit")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam!")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_faces_encodings, encoding, tolerance=0.5)
                name = "Unknown Person"

                distances = face_recognition.face_distance(known_faces_encodings, encoding)
                best = np.argmin(distances)
                if matches[best]:
                    name = known_faces_names[best]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.rectangle(frame, (left, bottom - 40), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 10, bottom - 10),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Live Face Recognition - Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # =============== QUIT ===============
    elif choice == 'q':
        print("Bye bro! Rasengan complete")
        break

    else:
        print("Invalid input! Type 1, 2, or Q")
