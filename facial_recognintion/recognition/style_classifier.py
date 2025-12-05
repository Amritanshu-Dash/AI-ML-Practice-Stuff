#classifies real and anime faces...
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from config import RESULTS_FOLDER

class RealVsAnimeClassifier:
  def __init__(self):
    self.model_path = RESULTS_FOLDER / "real_vs_anime_classifier.pkl"
    self.model = None

    if os.path.exists(self.model_path):
      print("loading your secret weapon: REAL vs ANIME brain....")
      self.model = joblib.load(self.model_path)
      print(" RealVsAnimeClassifier loaded.")
    else:
      print("Can't load RealVsAnimeClassifier.")

    def extract_magic_numbers(self, embedding):
    # These 5 numbers are like "DNA fingerprint" for real vs drawn faces
      norm = np.linalg.norm(embedding) #total power of the face dna =, matrix determinant value
      std = np.std(embedding) #calculates the standard deviation of the matrix or vector of the image to see how much numbers jump around
      mean = np.mean(embedding) #the average value of the matrix
      skewness = np.mean((embedding - mean)**3)/(std**3 + 1e-8)
      kurtosis = np.mean((embedding - mean)**4)/(std**4 + 1e-8)
      return [norm, std, mean, skewness, kurtosis]

    def predict(self, embedding):
      #check is the pretrained model is ready to use or not
      if self.model is not None:
        features = self.extract_magic_number(embedding)
        pred = self.model.predict([features])[0]
        prob = self.model.predict_prob([features])[0].max()
        return "real" if pred == 1 else "anime", prob
      else:
        #if pre trained model is not working then we are using this
        strength = np.linalg.norm(embedding)
        if strength < 0.95:
          return "anime", 0.88
        else:
          return "real", 0.93

    def train_secret_weapon(self, real_faces, anime_faces):
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
      joblib.dump(model, self.model_path)
      print(f"Model is ready for real vs anime face difference → {self.model_path}")
      self.model = model

#make sure only 1 instance of the model every exist no matter how many number of times you call
_classifier = None
def get_style_classifier():
  global _classifier
  if _classifier is None:
    _classifier = RealVsAnimeClassifier()
  return _classifier
