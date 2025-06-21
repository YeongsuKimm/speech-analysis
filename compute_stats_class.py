import os
import numpy as np
import h5py
from classification import get_dict
from torch.utils.data import Dataset
import torch


class StreamerDatasetRaw(Dataset):
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
            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_")])
            label = self.label_dict[streamer]

            if self.mode == "both":
                for text_file, audio_file in zip(text_files, audio_files):
                    text_path = os.path.join(streamer_path, text_file)
                    audio_path = os.path.join(streamer_path, audio_file)
                    self.data.append((text_path, audio_path))
            elif self.mode == "text":
                for text_file in text_files:
                    text_path = os.path.join(streamer_path, text_file)
                    self.data.append((text_path,))
            elif self.mode == "audio":
                for audio_file in audio_files:
                    audio_path = os.path.join(streamer_path, audio_file)
                    self.data.append((audio_path,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "both":
            text_path, audio_path = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text = np.array(f['tensor']).squeeze(0)
            with h5py.File(audio_path, 'r') as f:
                audio = np.array(f['tensor']).squeeze(0)
            return text, audio

        elif self.mode == "text":
            (text_path,) = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text = np.array(f['tensor']).squeeze(0)
            return text

        elif self.mode == "audio":
            (audio_path,) = self.data[idx]
            with h5py.File(audio_path, 'r') as f:
                audio = np.array(f['tensor']).squeeze(0)
            return audio


def compute_and_save_stats(dataset, mode):
    all_features = []

    for i in range(len(dataset)):
        item = dataset[i]
        if mode == "both":
            text, audio = item
            all_features.append(("text", text))
            all_features.append(("audio", audio))
        else:
            all_features.append((mode, item))

    for mod in ["text", "audio"]:
        mod_feats = [feat for label, feat in all_features if label == mod]
        mod_feats = np.stack(mod_feats)
        mean = np.mean(mod_feats, axis=0)
        std = np.std(mod_feats, axis=0)
        np.save(f"norm_params/{mod}_class_mean.npy", mean)
        np.save(f"norm_params/{mod}_class_std.npy", std)
        print(f"{mod.upper()} mean/std saved: shape = {mean.shape}")


if __name__ == "__main__":
    mode = "both"
    root_dir = "processed"
    label_dict = get_dict(os.listdir(root_dir))
    dataset = StreamerDatasetRaw(root_dir=root_dir, label_dict=label_dict, mode=mode)
    compute_and_save_stats(dataset, mode)
