import face_recognition
import cv2
import numpy as np
import os

print("Looking for faces in known faces directory...")

# creating the directory for known faces if it doesn't exist
# ------------------- Folders -------------------
known_people_folder = "known_people"      # Naruto, Elon photos go here
recognising_folder = "recognising"         # Photos you want to check go here
results_folder = "results"                 # Where output photos will be saved

# Create folders if they don't exist
os.makedirs(known_people_folder, exist_ok=True)
os.makedirs(recognising_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

known_faces_encodings = []
known_faces_names = []

if len(os.listdir(known_people_folder)) == 0:
    print("Put photos in 'known_people' folder!")
    input("Press Enter to exit...")
    exit()

# Loop through each file in the known faces directory
for filename in os.listdir("known_faces"):
  if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    path = os.path.join("known_people", filename)
    print(f"Processing file: path => {path} and filename => {filename}")
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image) ## Get the face encodings DNA of the face

    if len(encoding) > 0:
      known_faces_encodings.append(encoding[0])
      name = os.path.splitext(filename)[0] # Get the name without the file extension that means if file is filname.jpg it will return filename
      known_faces_names.append(name)
      print(f"Successfully added encoding for {name}")
    else:
      print(f"No faces found in the image {filename}, skipping...")

#if no known faces were found, tell the user
if (len(known_faces_encodings) == 0):
  print("No known faces found. Please add clear images of known faces to the 'known_faces' directory.")
  input("Press Enter to exit...")
  exit()

else :
  print(f"Found {len(known_faces_encodings)} known faces and name are {known_faces_names}.")

# ------------------- Menu -------------------
while True:
    print("═" * 50)
    print("       NARUTO & ELON DETECTOR PRO")
    print("═" * 50)
    print("Press 1 → Check all photos in 'recognising' folder")
    print("Press 2 → Live Webcam Mode")
    print("Press Q → Quit")
    print("═" * 50)
    choice = input("What do you want to do? (1/2/Q): ").strip().lower()

    if choice == '1':
      print("\nScanning all photos in 'recognising' folder...")
      photos = [f for f in os.listdir(recognising_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

      if len(photos) == 0:
        print("No photos found in 'recognising' folder. Please add photos to check.")
        input("Press Enter to return to continue...")
        continue

      for photo in photos:
        path = os.path.join(recognising_folder, photo)
        print(f"Processing file path: {path} and photo: {photo}")

        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        encoding = face_recognition.face_encodings(image, locations)

        img = cv2.imread(path)

        for(top, right, bottom, left), encoding in zip(locations, encoding):
          matches = face_recognition.compare_faces(known_faces_encodings, encoding, tolerance=0.5)
          name = "Unknown Person"
          distances = face_recognition.face_distance(known_faces_encodings, encoding)
          best = np.argmin(distances)

          if matches[best]:
            name = known_faces_names[best]
            print(f"Found {name} in {photo}!")

          #draw rectangle around the face and name
          cv2.rectangle(img, name, (left, top), (right, bottom), (0, 255, 0), 4)
          cv2.rectangle(img, (left, bottom - 50), (right, bottom), (0, 255, 0), cv2.FILLED)
          cv2.putText(img, name, (left + 10, bottom - 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

          #save result image
          output_path = os.path.join(results_folder, "RESULT_" + photo)
          cv2.imwrite(output_path, img)
          print(f"Result saved to {output_path}")

    print("Done processing all photos. check the 'results' folder for output images.")
    input("Press Enter to return to the menu...")

    elif choice == '2':
      print("\nStarting live webcam mode...")
      cap = cv2.VideoCapture(0)
      cap.set(3, 1000)
      cap.set(4, 800)
      while True:
       ret.friend()
