import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from classification import get_dict
import h5py
import numpy as np


# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mode selection: "text", "audio", or "both"
MODE = "both"

# Define the Streamer Dataset (supports all modes)
class StreamerDataset(Dataset):
    def __init__(self, root_dir, label_dict, mode="both", normalize=True):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.label_dict = label_dict
        self.mode = mode
        self.normalize = normalize
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
        if normalize:
            if mode in ["text", "both"]:
                self.mean_text = torch.tensor(np.load("norm_params/text_class_mean.npy"), dtype=torch.float32)
                self.std_text = torch.tensor(np.load("norm_params/text_class_std.npy"), dtype=torch.float32)
            if mode in ["audio", "both"]:
                self.mean_audio = torch.tensor(np.load("norm_params/audio_class_mean.npy"), dtype=torch.float32)
                self.std_audio = torch.tensor(np.load("norm_params/audio_class_std.npy"), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if MODE == "both":
            text_path, audio_path, label = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            with h5py.File(audio_path, 'r') as f:
                audio_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            
            if self.normalize:
                text_features = (text_features - self.mean_text) / (self.std_text + 1e-8)
                audio_features = (audio_features - self.mean_audio) / (self.std_audio + 1e-8)
            
            label = torch.tensor(label, dtype=torch.long)
            return text_features, audio_features, label

        elif MODE == "text":
            text_path, label = self.data[idx]
            with h5py.File(text_path, 'r') as f:
                text_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            if self.normalize:
                text_features = (text_features - self.mean_text) / (self.std_text + 1e-8)
            label = torch.tensor(label, dtype=torch.long)
            return text_features, label

        elif MODE == "audio":
            audio_path, label = self.data[idx]
            with h5py.File(audio_path, 'r') as f:
                audio_features = torch.tensor(np.array(f['tensor']), dtype=torch.float32).squeeze(0)
            if self.normalize:
                audio_features = (audio_features - self.mean_audio) / (self.std_audio + 1e-8)
            label = torch.tensor(label, dtype=torch.long)
            return audio_features, label


# Model Definitions
class TextOnlyMLP(nn.Module):
    def __init__(self, text_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class AudioOnlyMLP(nn.Module):
    def __init__(self, audio_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiModalMLP(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, output_dim):
        super().__init__()
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_input, audio_input):
        text_feat = self.text_mlp(text_input)
        audio_feat = self.audio_mlp(audio_input)
        fused = torch.cat((text_feat, audio_feat), dim=1)
        return self.fusion_mlp(fused)

# Training + Validation
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    model.to(DEVICE)

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if MODE == "both":
                text, audio, labels = batch
                text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
                outputs = model(text, audio)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total * 100
        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if MODE == "both":
                    text, audio, labels = batch
                    text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
                    outputs = model(text, audio)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    # Parameters
    TEXT_DIM = AUDIO_DIM = 768
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LR = 1e-2

    streamer_list = os.listdir("processed")
    label_dict = get_dict(streamer_list)
    dataset = StreamerDataset(root_dir="processed", label_dict=label_dict, mode=MODE, normalize=True)

    # Split into train and validation sets
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if MODE == "text":
        model = TextOnlyMLP(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "text_only_class_model_normalized_t70-2.pth"
    elif MODE == "audio":
        model = AudioOnlyMLP(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "audio_only_class_model_normalized_t70-2.pth"
    else:
        model = MultiModalMLP(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "streamer_class_model_normalized_t70-2.pth"
    print(model_name)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=NUM_EPOCHS)

    model_path = f"models/{model_name}"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
