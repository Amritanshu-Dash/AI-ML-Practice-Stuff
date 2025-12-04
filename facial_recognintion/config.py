# config.py
import os
from pathlib import Path

# ===================================================================
# MAIN PATHS — CHANGE ONLY IF YOU WANT CUSTOM LOCATIONS
# ===================================================================
BASE_DIR = Path(__file__).parent.resolve()

KNOWN_PEOPLE_FOLDER = BASE_DIR / "known_people"
TEST_FOLDER = BASE_DIR / "test_folder"
RESULTS_FOLDER = BASE_DIR / "results"

# Subclasses for organized results
RESULT_SUBFOLDERS = [
    "Elon_Musk",
    "Naruto",
    "Other_Real_Humans",
    "Other_Anime",
    "Unknown"
]

# ===================================================================
# MODEL & RECOGNITION SETTINGS
# ===================================================================
INSIGHTFACE_MODEL = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.55   # Lower = stricter (0.4–0.6 is good range)
ANIME_NORM_THRESHOLD = 0.92   # Fallback if no SVM: anime embeddings usually have lower norm

# Confidence = 1 - distance (normalized)
CONFIDENCE_DISPLAY = lambda dist: max(0, 1 - (dist / 1.2))

# Supported image formats
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Excel log files
TRAINING_LOG_FILE = RESULTS_FOLDER / "Training_Log.xlsx"
TESTING_LOG_FILE = RESULTS_FOLDER / "Testing_Log.xlsx"

# ===================================================================
# AUTO CREATE FOLDERS
# ===================================================================
for folder in [KNOWN_PEOPLE_FOLDER, TEST_FOLDER, RESULTS_FOLDER]:
    folder.mkdir(exist_ok=True)

for sub in RESULT_SUBFOLDERS:
    (RESULTS_FOLDER / sub).mkdir(exist_ok=True)

print("✅ Config loaded & folders ready!")
print(f"   Training → {KNOWN_PEOPLE_FOLDER}")
print(f"   Testing  → {TEST_FOLDER}")
print(f"   Results  → {RESULTS_FOLDER}")
