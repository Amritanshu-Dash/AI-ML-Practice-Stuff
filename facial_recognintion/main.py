# main.py
# Entry point of the entire system
# Run this file: python main.py

import os
import cv2
from model.face_analyzer import get_analyzer
from training.trainer import train_all
from recognition.recognizer import recognize_face, load_known_embeddings
from utils.excel_writer import log_testing
from config import TEST_FOLDER,RESULTS_FOLDER ,RESULT_SUBFOLDERS

def main_menu():

  print("\n" + "="*70)
  print("     Real + Anime Face Recognition System")
  print("="*70)

  while True:
    print("\n1. Train on known people (known_people folder)")
    print("2. Test all folders in test_folder")
    print("3. Live webcam detection")
    print("Q. Quit")
    choice = input("\nSelect option (1/2/3/Q): ").strip().lower()

    if choice == "1":
      train_all()
      input("\nTraining finished. Press Enter to continue...")
    elif choice == "2":
      run_batch_test()
    elif choice == "3":
      run_live_webcam()
    elif choice in ["q", "quit"]:
      print("\nGoodbye!")
      break
    else:
      print("Invalid option. Try again.")

def run_batch_test():

  known = load_known_embeddings()

  if not known:
    print("No trained data, Train first")
    return
  analyzer = get_analyzer()
  all_results = []

  print("\nStarting batch testing...")

  for folder_name in os.listdir(TEST_FOLDER):

    folder_path = os.path.join(TEST_FOLDER, folder_name)
    if not os.path.isdir(folder_path):
      continue

    print(f"\nTesting → {folder_name}")
    correct = total = 0

    for file in os.listdir(folder_path):

      if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)
        if img is None:
          continue
        faces = analyzer.get_faces(img)
        if not faces:
          detected = "No Face"
          conf = 0.0
        else:
          detected, conf = recognize_face(faces[0].embedding, known)

        # Simple accuracy check
        true_label = folder_name.replace("test_", "").replace("-", " ").title()
        is_correct = detected.lower() in true_label.lower() or true_label.lower() in detected.lower()

        if is_correct:
          correct += 1
        total += 1

        # Save image to correct results subfolder
        subfolder = detected.replace(" ", "_")
        if subfolder not in RESULT_SUBFOLDERS:
          subfolder = "Unknown"
        save_dir = RESULTS_FOLDER / subfolder
        save_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(save_dir/f"RESULT_{file}"), img)

        all_results.append({
          "Filename": file,
          "Folder": folder_name,
          "Detected_As": detected,
          "Confidence": conf,
          "Correct?": is_correct
        })

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  → {correct}/{total} correct ({accuracy:.1f}%)")

  log_testing(all_results, "Batch_Test")
  print(f"\nAll results saved in: {RESULTS_FOLDER}")

def run_live_webcam():
  known = load_known_embeddings()
  if not known:
    print("No trained data. Train first.")
    return
  analyzer = get_analyzer()
  cap = cv2.VideoCapture(0)
  print("\nWebcam started. Press 'q' to quit." )

  while True:

    ret, frame = cap.read()
    if not ret:
      break

    faces = analyzer.get_faces(frame)

    for face in faces:
      name, conf = recognize_face(face.embedding, known)
      bbox = face.bbox.astype(int)

      color = (0, 255, 0) if "Unknown" not in name and "Other" not in name else (0, 255, 255)

      cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
      label = f"{name} ({conf:.2f})"
      cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face recognition - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main_menu()
