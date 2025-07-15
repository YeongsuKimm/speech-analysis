import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from separateClass import MultiModalMLP, TextOnlyMLP, AudioOnlyMLP 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "audio"  # change as needed: "text", "audio", or "both"
BATCH_SIZE = 16

# Load your label dictionary function or data
from classification import get_dict

# Normalization flags
NORMALIZE = True

class Evaluator:
    def __init__(self, model, criterion, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device

    def evaluate(self, data_loader, mode="both"):
        """
        Evaluate model on the data_loader.

        mode: "text", "audio", or "both" to match your dataset and model inputs.
        
        Returns:
            avg_loss (float): average loss over dataset
            accuracy (float): accuracy in percentage
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                if mode == "both":
                    text, audio, labels = batch
                    text, audio, labels = text.to(self.device), audio.to(self.device), labels.to(self.device)
                    outputs = self.model(text, audio)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                print(labels)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                print(preds)

        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        return avg_loss, accuracy

class StreamerDataset(Dataset):
    def __init__(self, root_dir, label_dict, mode="both", normalize=True):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.label_dict = label_dict
        self.mode = mode
        self.normalize = normalize

        # Load normalization stats if needed
        if normalize:
            if mode in ["text", "both"]:
                self.mean_text = torch.tensor(np.load("norm_params/text_class_mean.npy"), dtype=torch.float32)
                self.std_text = torch.tensor(np.load("norm_params/text_class_std.npy"), dtype=torch.float32)
            if mode in ["audio", "both"]:
                self.mean_audio = torch.tensor(np.load("norm_params/audio_class_mean.npy"), dtype=torch.float32)
                self.std_audio = torch.tensor(np.load("norm_params/audio_class_std.npy"), dtype=torch.float32)


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
                    self.data.append((text_path, audio_path, label))
            elif self.mode == "text":
                for text_file in text_files:
                    text_path = os.path.join(streamer_path, text_file)
                    self.data.append((text_path, label))
            elif self.mode == "audio":
                for audio_file in audio_files:
                    audio_path = os.path.join(streamer_path, audio_file)
                    self.data.append((audio_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "both":
            text_path, audio_path, label = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            with h5py.File(audio_path, 'r') as f:
                audio_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)

            if self.normalize:
                text_features = (text_features - self.mean_text) / self.std_text
                audio_features = (audio_features - self.mean_audio) / self.std_audio
            label = torch.tensor(label, dtype=torch.long)
            return text_features, audio_features, label

        elif self.mode == "text":
            text_path, label = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            if self.normalize:
                text_features = (text_features - self.mean_text) / self.std_text
            label = torch.tensor(label, dtype=torch.long)
            return text_features, label

        elif self.mode == "audio":
            audio_path, label = self.data[idx]
            with h5py.File(audio_path, 'r') as f:
                audio_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            if self.normalize:
                audio_features = (audio_features - self.mean_audio) / self.std_audio
            # print(audio_path)
            label = torch.tensor(label, dtype=torch.long)
            return audio_features, label


def load_model(mode, device):
    TEXT_DIM = AUDIO_DIM = 768  # adjust if needed
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3

    if mode == "text":
        model = TextOnlyMLP(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "models/text_only_class_model_normalized_t70-3.pth"
    elif mode == "audio":
        model = AudioOnlyMLP(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "models/audio_only_class_model_normalized_t70-3.pth"
    else:
        model = MultiModalMLP(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "models/streamer_class_model_normalized_t70-3.pth"

    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    streamer_list = os.listdir(".test")
    label_dict = get_dict(streamer_list)

    dataset = StreamerDataset(root_dir=".test", label_dict=label_dict, mode=MODE, normalize=NORMALIZE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(MODE, DEVICE)
    criterion = nn.CrossEntropyLoss()

    evaluator = Evaluator(model, criterion, DEVICE)
    loss, accuracy = evaluator.evaluate(data_loader, mode=MODE)

    print(f"Evaluation Loss: {loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
