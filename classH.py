import os
import h5py
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from classification import get_dict

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mode selection: "text", "audio", or "both"
MODE = "audio"

# Define the Streamer Dataset (supports all modes)
class StreamerDataset(Dataset):
    def __init__(self, root_dir, label_dict, mode="both"):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.label_dict = label_dict
        self.mode = mode

        self.data = []
        for streamer in self.streamers:
            if streamer not in self.label_dict:
                continue

            streamer_path = os.path.join(root_dir, streamer)
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_")])
            label = self.label_dict[streamer]

            for audio_file in audio_files:
                audio_path = os.path.join(streamer_path, audio_file)
                self.data.append((audio_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        audio_features = torch.load(audio_path).squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return audio_features, label
    

if __name__ == "__main__":
    BATCH_SIZE = 1
    H5_FILENAME = "class_dataset.h5"
    streamer_list = os.listdir("processed_copy")
    label_dict = get_dict(streamer_list)
    dataset = StreamerDataset(root_dir="processed_copy", label_dict=label_dict, mode=MODE) 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features = []
    all_labels = []

    for features, labels in tqdm(loader):
        all_features.append(features.squeeze(0))  # Shape: [feature_dim]
        all_labels.append(labels.item())          # Convert 0-dim tensor to Python int

    # Convert to single tensor
    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Save to H5
    with h5py.File(H5_FILENAME, "w") as hf:
        hf.create_dataset("features", data=features_tensor.cpu().numpy())
        hf.create_dataset("labels", data=labels_tensor.numpy())

    print(f"Saved dataset with {len(dataset)} samples to {H5_FILENAME}")