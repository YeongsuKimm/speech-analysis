import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mode: "text", "audio", or "both"
MODE = "text"

class StreamerDataset(Dataset):
    def __init__(self, root_dir, mode="both", norm_dir=None):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.mode = mode
        self.data = []
        
        self.norm_params = {}
        if norm_dir:
            if mode in ["both", "text"]:
                self.norm_params['text_mean'] = torch.tensor(np.load(f"{norm_dir}/text_reg_mean.npy"), dtype=torch.float32)
                self.norm_params['text_std'] = torch.tensor(np.load(f"{norm_dir}/text_reg_std.npy"), dtype=torch.float32)
            if mode in ["both", "audio"]:
                self.norm_params['audio_mean'] = torch.tensor(np.load(f"{norm_dir}/audio_reg_mean.npy"), dtype=torch.float32)
                self.norm_params['audio_std'] = torch.tensor(np.load(f"{norm_dir}/audio_reg_std.npy"), dtype=torch.float32)
            if os.path.exists(f"{norm_dir}/label_mean.npy") and os.path.exists(f"{norm_dir}/label_std.npy"):
                self.norm_params['label_mean'] = torch.tensor(np.load(f"{norm_dir}/label_mean.npy"), dtype=torch.float32)
                self.norm_params['label_std'] = torch.tensor(np.load(f"{norm_dir}/label_std.npy"), dtype=torch.float32)

        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            metadata_path = os.path.join(streamer_path, "metadata.h5")
            if not os.path.exists(metadata_path):
                continue
            with h5py.File(metadata_path, "r") as f:
                if "tensor" in f:
                    target = f["tensor"][2]
                else:
                    continue

            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_") and f.endswith(".h5")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_") and f.endswith(".h5")])

            if mode == "both":
                for text_file, audio_file in zip(text_files, audio_files):
                    self.data.append((os.path.join(streamer_path, text_file), os.path.join(streamer_path, audio_file), target))
            elif mode == "text":
                for text_file in text_files:
                    self.data.append((os.path.join(streamer_path, text_file), target))
            elif mode == "audio":
                for audio_file in audio_files:
                    self.data.append((os.path.join(streamer_path, audio_file), target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "both":
            text_path, audio_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            audio_feat = self.load_h5_features(audio_path)

            # Normalize features if stats are loaded
            if 'text_mean' in self.norm_params and 'text_std' in self.norm_params:
                text_feat = (text_feat - self.norm_params['text_mean']) / self.norm_params['text_std']
            if 'audio_mean' in self.norm_params and 'audio_std' in self.norm_params:
                audio_feat = (audio_feat - self.norm_params['audio_mean']) / self.norm_params['audio_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return text_feat, audio_feat, target

        elif self.mode == "text":
            text_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            if 'text_mean' in self.norm_params and 'text_std' in self.norm_params:
                text_feat = (text_feat - self.norm_params['text_mean']) / self.norm_params['text_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return text_feat, target

        elif self.mode == "audio":
            audio_path, target = self.data[idx]
            audio_feat = self.load_h5_features(audio_path)
            if 'audio_mean' in self.norm_params and 'audio_std' in self.norm_params:
                audio_feat = (audio_feat - self.norm_params['audio_mean']) / self.norm_params['audio_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return audio_feat, target

    def load_h5_features(self, file_path):
        with h5py.File(file_path, "r") as f:
            if "tensor" in f:
                data = f["tensor"][()]
        return torch.tensor(data, dtype=torch.float32).squeeze(0)
    
class TextOnlyRegressor(nn.Module):
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

class AudioOnlyRegressor(nn.Module):
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

class MultiModalRegressor(nn.Module):
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

# Training function
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device="cuda", mode="both"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if mode == "both":
                text, audio, target = batch
                text, audio, target = text.to(device), audio.to(device), target.to(device)
                output = model(text, audio).squeeze()
            else:
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs).squeeze()
            target = target.squeeze()
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if mode == "both":
                    text, audio, target = batch
                    text, audio, target = text.to(device), audio.to(device), target.to(device)
                    output = model(text, audio).squeeze()
                else:
                    inputs, target = batch
                    inputs, target = inputs.to(device), target.to(device)
                    output = model(inputs).squeeze()
                # print(output)
                target = target.squeeze()
                # print(target)
                loss = criterion(output, target)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# Main
if __name__ == "__main__":
    dataset_size = 127
    batch_size = 16
    epochs = 30
    lr = 1e-2

    os.makedirs("models", exist_ok=True)

    # Load dataset with normalization stats
    dataset = StreamerDataset("processed/", mode=MODE, norm_dir="norm_params")

    # Create train/val split based on streamer folders
    streamer_folders = sorted(os.listdir("processed/"))
    num_streamers = len(streamer_folders)

    train_idx = list(range(int(0.8 * num_streamers)))
    val_idx = list(range(int(0.8 * num_streamers), num_streamers))

    train_streamers = [streamer_folders[i] for i in train_idx]
    val_streamers = [streamer_folders[i] for i in val_idx]

    train_indices = [
        i for i, entry in enumerate(dataset.data)
        if any(s in entry[0] for s in train_streamers)
    ]
    val_indices = [
        i for i, entry in enumerate(dataset.data)
        if any(s in entry[0] for s in val_streamers)
    ]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Model selection
    TEXT_DIM = AUDIO_DIM = 768
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1

    if MODE == "text":
        model = TextOnlyRegressor(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "text_only_reg_model_normalized_t70-2.pth"
    elif MODE == "audio":
        model = AudioOnlyRegressor(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "audio_only_reg_model_normalized_t70-2.pth"
    else:
        model = MultiModalRegressor(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_name = "streamer_reg_model_normalized_t70-2.pth"
    print(model_name)
    # Train & save
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, mode=MODE)
    model_path = f"models/{model_name}"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
