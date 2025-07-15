import os
import json
import torch
import tqdm
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio.transforms as transforms
import h5py


DATA_DIR = "data"
SAVE_DIR ="processed"

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load frozen feature extractors ONCE
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base",output_hidden_states = True).to(DEVICE).eval()

# roberta_model.config.output_hidden_states = True;

wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = AutoModel.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True).to(DEVICE).eval()

def extract_text_features(text):
    tokens = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output = roberta_model(**tokens)
    return output.hidden_states[-2][:,0,:]  # Shape: [1, 768]

def extract_audio_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

    # Process input for Wav2Vec2
    inputs = wav2vec_processor(waveform.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = wav2vec_model(**inputs)
    return output.hidden_states[-2][:,0,:]  # Shape: [1, 768]

def process_streamer(streamer):
    print("Processing: " + streamer)
    streamer_path = os.path.join(DATA_DIR, streamer)
    json_path = os.path.join(streamer_path, "text_audio_pairs.json")
    metadata_path = os.path.join(streamer_path, f"{streamer}.txt")
    print(streamer_path)
    if not (os.path.exists(json_path) and os.path.exists(metadata_path)):
        return
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    # Load metadata
    metadata = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                key, value = line.strip().split(": ")
                metadata[key] = float(value)  # Convert numeric metadata
            except ValueError:
                metadata[key] = value
    try:
        current_subscribers = float(metadata.get("Current Subscribers", 0))
    except:
        current_subscribers = 0
    log_current_subscribers = max(0, np.log10(current_subscribers + 1))
    
    try:
        peak_subscribers = float(metadata.get("All-Time High Active Subscribers", 0))
    except:
        peak_subscribers = 0
    log_peak_subscribers = max(0, np.log10(peak_subscribers))

    try:
        followers = float(metadata.get("Total Followers", 0))
    except:
        followers = 0
    log_followers = max(0, np.log10(followers))

    # Save path
    save_path = os.path.join(SAVE_DIR, streamer)
    os.makedirs(save_path, exist_ok=True)

    # Process and save each text/audio pair
    for i, entry in enumerate(tqdm.tqdm((json_data))):
        text_file = entry["text"]
        audio_file = entry["audio"]

        # Load text
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        text_features = extract_text_features(text)  # [1, 768]

        # Load audio
        audio_features = extract_audio_features(audio_file)  # [1, 768]

        # Save tensors
        torch.save(text_features, os.path.join(save_path, f"text_{i}.pt"))
        torch.save(audio_features, os.path.join(save_path, f"audio_{i}.pt"))

    # Save metadata
    metadata_tensor = torch.tensor([log_current_subscribers, log_peak_subscribers, log_followers], dtype=torch.float32)
    torch.save(metadata_tensor, os.path.join(save_path, "metadata.pt"))

    print(f"Processed: {streamer}")


def process_streamer_metadata(streamer):
    streamer_path = os.path.join(DATA_DIR, streamer)
    metadata_path = os.path.join(streamer_path, f"{streamer}.txt")
    print(streamer_path)

    metadata = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                key, value = line.strip().split(": ")
                metadata[key] = float(value)  # Convert numeric metadata
            except ValueError:
                metadata[key] = value  

    try:
        current_subscribers = float(metadata.get("Current Subscribers", 0))
    except:
        current_subscribers = 0
    log_current_subscribers = max(0, np.log10(current_subscribers + 1))
    
    try:
        peak_subscribers = float(metadata.get("All-Time High Active Subscribers", 0))
    except:
        peak_subscribers = 0
    log_peak_subscribers = max(0, np.log10(peak_subscribers))

    try:
        followers = float(metadata.get("Total Followers", metadata.get("Followers", 0)))
    except:
        followers = 0
    log_followers = max(0, np.log10(followers))

    # Save path
    save_path = os.path.join(SAVE_DIR, streamer)
    os.makedirs(save_path, exist_ok=True)
    
    metadata_array = np.array([log_current_subscribers, log_peak_subscribers, log_followers], dtype=np.float32)
    
    h5_path = os.path.join(save_path, "metadata.h5")
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("tensor", data=metadata_array)

    print(log_followers)


# streamers = os.listdir(DATA_DIR)
# print(len(streamers))

# streamers = []
# import os
# for streamer in os.listdir("data/"):
#     streamers.append(streamer)

# for streamer in os.listdir("data"):
#     streamers.append(streamer)

# for streamer in streamers:
#     if streamer == "audio_conv.py":
#         continue
#     process_streamer_metadata(streamer)

streamers = []
with open("todo.txt", "r") as file:
    for line in file:
        streamers.append(line[:-1])
print(streamers)

batch_size = len(streamers) // 2
batch1 = streamers[:batch_size]
batch2 = streamers[batch_size:]

# Print results
print("Batch 1:", batch1)
print("Batch 2:", batch2)

completed = []
with open("completed.txt", "r") as file:
    for name in file:
        completed.append(name[:-1])

for streamer in streamers:
    if streamer in batch1 and streamer not in completed:
        process_streamer(streamer)
        with open("completed.txt", "a") as file:
            file.write(streamer + "\n")