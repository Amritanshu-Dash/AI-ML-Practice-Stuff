# download_datasets.py
# Purpose: Download high-quality face datasets automatically (run ONLY ONCE)
# Warning: Do not run multiple times — it will fill your storage with duplicate images
# If known_people/ already has photos → skip this file completely

import os               # To create folders and check files
import urllib.request   # To download files from the internet
import zipfile          # To extract .zip files
import tarfile          # To extract .tgz/.tar files (LFW datasets)
import shutil           # To copy/move/delete files and folders

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_FOLDER = "known_people"
TARGET_FOLDERS = ["elon_musk", "naruto", "other_real_names", "anime_characters"]

# create main folders if they don't exist
for folders in TARGET_FOLDERS:
  os.makedirs(os.path.join(BASE_FOLDER, folder), exist_ok=True)

# Check if the data already exists -> prevents re-download
if len(os.listdir(os.path.join(BASE_FOLDER, "elon_musk"))) > 20:
  print("Datasets already downloaded! Skipping...")
  print("   → If you want fresh data: delete the 'known_people' folder first")
  exit()

print("Starting dataset download (one-time only, ~2–8 minutes)...\n")

# ------------------------------------------------------------------
# 1. Elon Musk real photos (100+ high-quality images)
# ------------------------------------------------------------------

print("Downloading Elon Musk photos...")

url_elon = "https://github.com/italojs/facial-recognition-dataset/raw/master/elon_musk.zip"
urllib.request.urlretrieve(url_elon, "temp_elon.zip")
with zipfile.ZipFile("temp_elon.zip") as z:
  z.extractall(os.path.join(BASE_FOLDER, "elon_musk"))
os.remove("temp_elon.zip")
print("   Elon Musk photos → done")

# ------------------------------------------------------------------
# 2. Naruto + Anime character faces
# ------------------------------------------------------------------

print("Downloading Naruto & anime character dataset...")
url_anime = "https://github.com/nagadomi/animeface-character-dataset/raw/master/data.zip"

urllib.request.urlretrieve(url_anime, "temp_anime.zip")
with zipfile.ZipFile("temp_anime.zip") as z:
  z.extractall("temp_anime_folder")


# Separate Naruto form other Anime characters
for filename in os.listdir("temp_anime_folder"):
  filepath = os.path.join("temp_anime_folder", filename)
  if os.path.isfile(filepath):
    if "naruto" in filename.lower() or "uzumaki" in filename.lower():
      shutil.copy(filepath, os.path.join(BASE_FOLDER, "naruto"))
    else:
      shutil.copy(filepath, os.path.join(BASE_FOLDER, "anime_characters"))

shutil.rmtree("temp_anime_folder")
os.remove("temp_anime.zip")
print("   Naruto & anime faces → done")

# ------------------------------------------------------------------
# 3. Real humans from LFW (Labeled Faces in the Wild) dataset
# ------------------------------------------------------------------

print("Downloading real human faces (LFW dataset)...")
url_lfw = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
urllib.request.urlretrieve(url_lfw, 'lfw.tgz')

with tarfile.open("lfw.tgz") as t:
  t.extractall("temp_lfw")

count, target = 0, 500 #target limit is 500 to avoid too much storage usage

for person_name in os.listdir("temp_lfw/lfw"):
  person_path = os.path.join("temp_lfw/lfw", person_name)
  if os.path.isdir(person_path) and count < target:
    for img_file in os.listdir(person_path):
      if count >= target:
        break
      if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy(
          os.path.join(person_path, img_file),
          os.path.join(BASE_FOLDER, "other_real_humans")
        )
        count += 1

shutil.rmtree("temp_lfw")
os.remove("lfw.tgz")
print(f"   Real human faces → {count} images added")

# ------------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------------
print("\n" + "="*60)
print("ALL DATASETS DOWNLOADED SUCCESSFULLY!")
print("="*60)
print("You now have professional training data:")
print("   • elon_musk/           → Real person")
print("   • naruto/              → Specific anime character")
print("   • anime_characters/    → General anime faces")
print("   • other_real_humans/   → Random real people")
print("\nNext steps:")
print("   1. Run: python main.py")
print("   2. Choose option 1 → Train")
print("   3. Choose option 3 → Webcam → look at camera")
print("\nEnjoy the magic")
