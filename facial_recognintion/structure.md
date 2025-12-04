face_recognition_system/
│
├── main.py                     # Menu + launcher
├── config.py                   # All paths, thresholds, settings
├── model/
│   ├── face_analyzer.py        # Load InsightFace once ---
│   └── embedding_extractor.py  # Get embeddings + bbox
│
├── training/
│   ├── trainer.py              # Train all known people + general classes ---
│   └── known_embeddings.pkl    # Saved after training
│
├── recognition/
│   ├── recognizer.py           # Core matching logic (with real/anime detection)
│   └── style_classifier.py    # NEW: Detect if face is Real or Anime (critical!)
│
├── testing/
│   ├── batch_tester.py         # Test entire folders
│   └── webcam_tester.py        # Live detection
│
├── utils/
│   ├── logger.py               # Excel + console logging
│   ├── folder_manager.py       # Create result subfolders
│   └── excel_writer.py         # Generate Training_Log.xlsx & Testing_Log.xlsx
│
├── known_people/               # Training data
│   ├── elon_musk/
│   ├── naruto/
│   ├── other_real_humans/      # Random real people photos
│   └── anime_characters/       # Various anime (non-Naruto)
│
├── test_folder/                # Put test batches here
│   ├── test_elon/
│   ├── test_naruto/
│   └── test_random/
│
└── results/                    # Auto-generated with subfolders + logs
