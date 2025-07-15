import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import h5py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StreamerDatasetRaw(Dataset):
    def __init__(self, root_dir, mode="both"):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.mode = mode
        self.data = []

        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            metadata_path = os.path.join(streamer_path, "metadata.h5")
            if not os.path.exists(metadata_path):
                continue
            
            with h5py.File(metadata_path, "r") as f:
                if "tensor" in f:
                    target = f["tensor"][2]  # Load scalar or array
                else:
                    # Fallback or error
                    print(f"Warning: follower_count not found in {metadata_path}")
                    continue

            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_") and f.endswith(".h5")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_") and f.endswith(".h5")])

            if mode == "both":
                for text_file, audio_file in zip(text_files, audio_files):
                    text_path = os.path.join(streamer_path, text_file)
                    audio_path = os.path.join(streamer_path, audio_file)
                    self.data.append((text_path, audio_path, target))
            elif mode == "text":
                for text_file in text_files:
                    text_path = os.path.join(streamer_path, text_file)
                    self.data.append((text_path, target))
            elif mode == "audio":
                for audio_file in audio_files:
                    audio_path = os.path.join(streamer_path, audio_file)
                    self.data.append((audio_path, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "both":
            text_path, audio_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            audio_feat = self.load_h5_features(audio_path)
            target = torch.tensor([target], dtype=torch.float32)
            return text_feat, audio_feat, target

        elif self.mode == "text":
            text_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            target = torch.tensor([target], dtype=torch.float32)
            return text_feat, target

        elif self.mode == "audio":
            audio_path, target = self.data[idx]
            audio_feat = self.load_h5_features(audio_path)
            target = torch.tensor([target], dtype=torch.float32)
            return audio_feat, target

    def load_h5_features(self, file_path):
        with h5py.File(file_path, "r") as f:
            if "tensor" in f:
                data = f["tensor"][()]
        return torch.tensor(data, dtype=torch.float32).squeeze(0)
    
def compute_and_save_stats(dataset, mode="both", save_dir="norm_params"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    text_feats = []
    audio_feats = []

    for i in range(len(dataset)):
        if mode == "both":
            text, audio, _ = dataset[i]
            text_feats.append(text.numpy())
            audio_feats.append(audio.numpy())
        elif mode == "text":
            text, _ = dataset[i]
            text_feats.append(text.numpy())
        elif mode == "audio":
            audio, _ = dataset[i]
            audio_feats.append(audio.numpy())

    if text_feats:
        text_feats = np.stack(text_feats)
        text_mean = text_feats.mean(axis=0)
        text_std = text_feats.std(axis=0) + 1e-8
        np.save(os.path.join(save_dir, "fold_text_reg_mean.npy"), text_mean)
        np.save(os.path.join(save_dir, "fold_text_reg_std.npy"), text_std)
        print("Saved text mean and std.")
    if audio_feats:
        audio_feats = np.stack(audio_feats)
        audio_mean = audio_feats.mean(axis=0)
        audio_std = audio_feats.std(axis=0) + 1e-8
        np.save(os.path.join(save_dir, "fold_audio_reg_mean.npy"), audio_mean)
        np.save(os.path.join(save_dir, "fold_audio_reg_std.npy"), audio_std)
        print("Saved audio mean and std.")

def compute_and_save_label_stats(dataset, save_dir="norm_params"):
    targets = []

    for entry in dataset.data:
        if isinstance(entry[0], str) and "text" in entry[0] and "audio" in entry[1]:  # both
            targets.append(entry[2])
        else:
            targets.append(entry[1])  # text or audio mode

    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    label_mean = targets_tensor.mean()
    label_std = targets_tensor.std()

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "fold_label_mean.npy"), label_mean.numpy())
    np.save(os.path.join(save_dir, "fold_label_std.npy"), label_std.numpy())

if __name__ == "__main__":
    MODE = "both"
    dataset = StreamerDatasetRaw("processed/", mode=MODE)
    compute_and_save_stats(dataset, MODE)
    compute_and_save_label_stats(dataset)
