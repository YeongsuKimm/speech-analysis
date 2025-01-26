import whisper

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


# model = whisper.load_model("medium")
# result = model.transcribe("audio_samples/introduction.mp3")
# print(result["text"])

# Set the output directory for saving model file
# Optionally, set a GCP bucket location

import os
import tensorflow as tf  # Ensure TensorFlow is installed

# Set the output directory for saving the model file
OUTPUT_DIR = 'OUTPUT_DIR_NAME'  # Replace with your desired output directory name
DO_DELETE = False  # Set to True if you want to clear the directory

# (Optional) GCP bucket setup
USE_BUCKET = False  # Set to True if using a GCP bucket
BUCKET = 'BUCKET_NAME'  # Replace with your GCP bucket name

# Handle GCP bucket case
if USE_BUCKET:
    OUTPUT_DIR = f'gs://{BUCKET}/{OUTPUT_DIR}'
    from google.auth.transport.requests import Request
    from google.auth import default
    # Authenticate with GCP
    credentials, project = default()
    credentials.refresh(Request())

# Handle local output directory
if not USE_BUCKET:
    if DO_DELETE and os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)  # Delete the directory if it exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

print(f'***** Model output directory: {OUTPUT_DIR} *****')
