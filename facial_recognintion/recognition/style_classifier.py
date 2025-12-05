#classifies real and anime faces...
# Purpose: Tell if a face is from a REAL human photo or an ANIME/cartoon drawing
# Why we need this: InsightFace was trained only on real faces → anime faces look like noise to it
# Without this filter → Naruto can be recognized as Elon Musk (disaster!)
# This class fixes that problem.

import numpy as np
import joblib
import os
from sklearn.svm import SVC
from config import RESULTS_FOLDER

class RealVsAnimeClassifier:
  def __init__(self):
    # Path where we save the trained classifier (so we don't retrain every time)
    self.model_path = RESULTS_FOLDER / "real_vs_anime_classifier.pkl"
    self.model = None

    # If we already trained before → load the model
    if os.path.exists(self.model_path):
      print("loading your secret weapon: REAL vs ANIME brain....")
      self.model = joblib.load(self.model_path)
      print(" RealVsAnimeClassifier loaded.")
    else:
      print("Can't load RealVsAnimeClassifier.")

    def extract_magic_numbers(self, embedding):
      # Convert the 512-number embedding into 5 simple numbers
      # These 5 numbers are like "DNA fingerprint" for real vs drawn faces and behave differently for the anime and real face
      norm = np.linalg.norm(embedding) #total power of the face dna =, matrix determinant value
      std = np.std(embedding) #calculates the standard deviation of the matrix or vector of the image to see how much numbers jump around
      mean = np.mean(embedding) #the average value of the matrix
      skewness = np.mean((embedding - mean)**3)/(std**3 + 1e-8)  # Shape (asymmetry)
      kurtosis = np.mean((embedding - mean)**4)/(std**4 + 1e-8)  # Peak sharpness
      return [norm, std, mean, skewness, kurtosis]

    def predict(self, embedding):
      # Main function: returns "real" or "anime" + confidence
      if self.model is not None:
        # Use the trained SVM model (very accurate)
        features = self.extract_magic_number(embedding)
        pred = self.model.predict([features])[0] # 0 = anime, 1 = real
        prob = self.model.predict_prob([features])[0].max()
        return "real" if pred == 1 else "anime", prob
      else:
        # Simple fallback: based on testing, anime faces usually have smaller norm
        strength = np.linalg.norm(embedding)
        if strength < 0.95:
          return "anime", 0.88
        else:
          return "real", 0.93

    def train_secret_weapon(self, real_faces, anime_faces):
      # Train a small SVM classifier using real and anime face embeddings
      # Run this only once when you have enough data
      print("Building the model for anime vs real difference...")
      X, y = [], []

      for emb in real_faces:
        X.append(self.extract_magic_numbers(emb))
        y.append(1) #1 -> for human face
      for emb in anime_faces:
        X.append(self.extract_magic_numbers(emb))
        y.append(0) #0 -> for anime face

      model = SVC(kernel="rbf", probability=True, C=100)
      model.fit(X, y)

      # Save the trained model
      joblib.dump(model, self.model_path)
      print(f"Model is ready for real vs anime face difference → {self.model_path}")
      self.model = model

#make sure only 1 instance of the model every exist no matter how many number of times you call
# Global variable — only one classifier in the whole program
_classifier = None

def get_style_classifier():
  # Returns the same classifier every time (saves memory and time)
  global _classifier
  if _classifier is None:
    _classifier = RealVsAnimeClassifier()
  return _classifier
