# This file creates beautiful Excel reports so you can SEE your AI getting smarter every day

import pandas as pd
from datetime import datetime
from config import RESULTS_FOLDER, TRAINING_LOG_FILE, TESTING_LOG_FILE
import os

def log_training(training_data):
  """
    Called after training
    training_data = list of dictionaries → e.g. [{"Person/Class": "Elon Musk", "Photos Trained": 87}]
  """
  if not training_data:
    print("No training data to log!")
    return

  # Turn the data into a nice table
  df = pd.DataFrame(training_data)
  df["Training Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  # Only sort if column exists and has data
  if "Photos Trained" in df.columns and len(df) > 0:
    df = df.sort_values("Photos Trained", ascending=False)

  path = RESULTS_FOLDER / "Training_Log.xlsx"

  #save to excel and creates the file if it doesn't exists.
  try:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
      df.to_excel(writer, sheet_name="Training_History", index=False)
    print(f"Training log saved → {path}")
  except Exception as e:
    print(f"Could not save training log: {e}")

def log_testing(test_results, batch_name="Unknown_Test"):
  """
    Called after testing a folder
    test_results = list like:
    [
        {"Filename": "elon1.jpg", "Detected_As": "Elon Musk", "Confidence": 0.96, "Correct?": True},
        ...
    ]
  """
  if not test_results:
    print("No test results to log!")
    return

  df = pd.DataFrame(test_results)
  df["Test Data"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  #clean name for excel sheet
  sheet_name = batch_name.replace("test_", "").replace("_", " ").title()[:30] # [:30] makes sure max 31 character

  try:
    #if file already exists then it will append to it so it will open in 'a' mode if file doesn't exists it will open on 'w' that is write mode like it will create a file and then write to it.
    mode = 'a' if os.path.exists(TESTING_LOG_FILE) else 'w'
    with pd.ExcelWriter(TESTING_LOG_FILE, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
      df.to_excel(writer, sheet_name=sheet_name, index=False)

    #calculate accuracy
    if "Correct?" in df.columns:
      accuracy = df["Correct?"].mean()*100
      correct = df["Correct?"].sum()
      total = len(df)
      print(f"{batch_name.upper()} → {correct}/{total} correct → Accuracy: {accuracy:.2f}%")

    print(f"Testing report saved → Sheet: {sheet_name}")
  except Exception as e:
    print(f"Could not save testing log: {e}")
