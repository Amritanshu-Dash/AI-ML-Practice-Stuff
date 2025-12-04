import os
import cv2
import pickle
import numpy as np
from model.face_analyzer import get_analyzer
from config import KNOWN_PEOPLE_FOLDER, IMAGE_EXTENSIONS, RESULTS_FOLDER
from utils.excel_writer import log_training
# training/trainer.py
# This file reads all photos from "known_people" folder and teaches the system who is who

def train_all():
  # Step 1: Get the AI brain (it will load only once thanks to our magic above)
  analyzer = get_analyzer()

  # This dictionary will store everyone's face DNA
  # Example: {"elon_musk": [dna1, dna2, dna3], "naruto": [dna4, dna5]}
  know_embeddings = {}

  # This list will store info for Excel log
  training_log = []

  print("\n" + "="*70)
  print("TRAINING PHASE: Processing known people images...")
  print("="*70 + "\n")

  # Loop through every folder inside known_people (elon_musk, naruto, etc.)
  for person_folder in os.listdir(KNOWN_PEOPLE_FOLDER):

    folder_path = os.path.join(KNOWN_PEOPLE_FOLDER, person_folder)

    # Skip if it's not a folder (like .DS_Store on Mac)
    if not os.path.isdir(folder_path):
      continue

    # Make name clean: "Elon Musk" → "elon_musk"
    person_name = person_folder.lower().replace(" ", "_")
    know_embeddings[person_name] = []
    print(f"Training on: {person_name.upper()}")

    successful_photos_count = 0


    for file in os.listdir(folder_path):
      if file.lower().endswith(IMAGE_EXTENSIONS): # Only image files
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
          print(f"  ⚠️  Warning: Could not read image {img_path}. Skipping.")
          continue

        # Ask the AI brain: "Are there any faces in this photo?"
        faces = analyzer.get_faces(img)

        if len(faces) > 0:
          # Take the first (and usually only) face
          embedding = faces[0].embedding # This is a list of 512 magic numbers
          know_embeddings[person_name].append(embedding)
          successful_photos_count += 1
          print(f"  ✅ Processed image: {file}")
        else:
          print(f"  ⚠️  No face detected in image: {file}. Skipping.")

    # Save how many photos we successfully trained for this person
    training_log.append({
      "Person/Class": person_name.replace("_", " ").title(),
      "Images Processed": successful_photos_count
    })

    if successful_photos_count == 0:
      print(f"  ⚠️  No valid images found for {person_name.upper()}.")
    else:
      print(f"  ➡️  Total images processed for {person_name.upper()}: {successful_photos_count}\n")


  # ==================== SAVE EVERYTHING ====================
  # Save the face DNA database so we don't retrain every time
  # Save embeddings to a file

  embedding_path = RESULTS_FOLDER / "known_embeddings.pkl"
  with open(embedding_path, "wb") as f:
    pickle.dump(know_embeddings, f)

  print(f"\n✅ Training completed! Embeddings saved to {embedding_path}\n")

  # Log training details to Excel
  # Create beautiful Excel log
  log_training(training_log)

  print("\n"+"="*70)
  print("Training complete and its summary:")

  # Final summary
  for name, embs in know_embeddings.items():
    print(f"   → {name.upper():20} : {len(embs)} photos")

  print(f"   Embeddings saved → {embedding_path}")
  print("="*70)

  return know_embeddings
