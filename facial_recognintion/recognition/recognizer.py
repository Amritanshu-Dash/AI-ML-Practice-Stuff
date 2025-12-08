# recognition/recognizer.py
# Purpose: Take a face embedding and return the person's name (or a category)
# It combines:
#   1. Style classification (real human vs anime) → from style_classifier.py
#   2. Similarity matching against known people
# Final output: name + confidence score

import numpy as np
import pickle
import os
from config import SIMILARITY_THRESHOLD, RESULTS_FOLDER
from recognition.style_classifier import get_style_classifier

def load_known_embeddings():
  """Load the dictionary of trained faces from disk."""
  path = RESULTS_FOLDER / "known_embeddings.pkl"
  if not os.path.exists(path):
    print("No trained data found. Please run training first.")
    return {}
  with open(path, "rb") as f:
    return pickle.load(f)

def recognize_face(face_embedding, known_embeddings=None):
  """
    Main recognition function.
    Input: one face embedding (512 numbers)
    Output: (name_string, confidence_float)
  """
  # Load known faces if not provided
  if known_embeddings is None:
    known_embeddings = load_known_embeddings()
    if not known_embeddings:
      return "Unknown", 0.0

  # Step 1: Decide if the face is real or anime
  classifier = get_style_classifier()
  style, style_confidence = classifier.predict(face_embedding)

  #step 2: Prepare candidate list based on style
  if style == "anime":
    # Only compare with anime people
    #basically in this for loop comprehension we are basically looping through dictionary known_embeddings and k and v are its key and value and we are checking is k is there or not
    candidates = {k : v for k, v in known_embeddings.items() if "naruto" in k.lower() or "anime" in k.lower()}
    fallback_name = "Other Anime"
  else:
    # Only compare with real people
    candidates = {K : v for k, v in known_embeddings.items() if "naruto" not in k.lower() or "anime" not in k.lower()}
    fallback_name = "Other Real Human"

  # If no specific candidates, use all known people
  if not candidates:
    candidates = known_embeddings

  # Step 3: Find the closest match
  best_distance = float('inf')
  best_name = "Unknown"

  for name, embedding_list in candidates.items():
    for known_emb in embedding_list:
      distance = np.linalg.norm(face_embedding - known_emb)
      if distance < best_distance:
        best_distance = distance
        best_name = name.replace("_"," ").title()

  # Step 4: Convert distance to confidence (0 to 1)
  confidence = max(0.0, 1.0 - (best_distance / 1.2))

  # Step 5: Apply threshold
  if best_distance < SIMILARITY_THRESHOLD:
    return best_name, round(confidence, 3)
  else:
    return fallback_name, round(confidence, 3)

  
