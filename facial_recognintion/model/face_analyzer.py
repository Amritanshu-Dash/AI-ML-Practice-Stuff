from insightface.app import FaceAnalysis
from config import INSIGHTFACE_MODEL, DETECTION_SIZE
import cv2
# model/face_analyzer.py
# This file is like the "brain loader" — it loads the heavy AI model ONCE and reuses it forever
# So we don't waste 3 seconds every time we want to detect a face

class FaceAnalyzer:

    # This is a magic trick called "Singleton Pattern"
    # Meaning: Only ONE brain (model) will ever exist in the whole program
    # Even if we call FaceAnalyzer() 1000 times, it gives the same brain back

  _instance = None # ← This is like a box that holds the brain. Right now, box is empty (None)

  def __new__(cls):

    # This function runs when someone says: analyzer = FaceAnalyzer()
    # "cls" means "the class itself" (FaceAnalyzer), not an object yet

    if cls._instance is None:
      # First time someone asks for the brain → box is empty → we need to create it
      print("/n InsightFace Model (buffalo_l) loading...")
      # Create a new "instance" (a real working object) using Python's super magic
      cls._instance = super(FaceAnalyzer, cls).__new__(cls)

      # Now we actually load the AI model into this object
      cls._instance.app = FaceAnalysis(
        name=INSIGHTFACE_MODEL, # "buffalo_l" = the best free model in 2025
        providers=['CPUExecutionProvider'] # ← This tells the model: "Run on CPU", But on Apple M1/M2 Macs, it secretly uses GPU automatically!
      )

      # Prepare the model (warm it up so it's ready to detect faces fast)
      cls._instance.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
      print("✅ InsightFace Model loaded!")

    # Whether it's first time or 100th time — return the SAME brain
    return cls._instance

  def get_faces(self, image):
    """Input: BGR image (from cv2). Output: List of face with .bbox and .embedding ."""
    # Give it an image (from cv2), it returns all faces with their DNA (embedding)
    return self.app.get(image)

#Global accessor
def get_analyzer():
  return FaceAnalyzer()
