import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio.transforms as transforms


# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load frozen feature extractors ONCE
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base",output_hidden_states = True).to(DEVICE).eval()

# roberta_model.config.output_hidden_states = True;


wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True).to(DEVICE).eval()

class StreamerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)[:70]

        self.data = []
        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            json_path = os.path.join(streamer_path, "text_audio_pairs.json")
            metadata_path = os.path.join(streamer_path, f"{streamer}.txt")  

            if os.path.exists(json_path) and os.path.exists(metadata_path):
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                
                metadata = self.load_metadata(metadata_path)
                
                followers = float(metadata.get("Followers", 0))
                if followers > 0:
                    log_followers = int(np.log10(followers + 1))-1  # log(followers + 1)
                for entry in json_data:
                    entry["metadata"] = metadata  # Add the full metadata to the entry
                    entry["metadata"]["Followers"] = log_followers  # Replace original followers with log-scaled value
                    self.data.append({"text": entry["text"], "audio": entry["audio"], "metadata": entry["metadata"]})

    def load_metadata(self, metadata_path):
        metadata = {}
        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    key, value = line.strip().split(": ")
                    metadata[key] = float(value)  # Convert all metadata values to float
                except ValueError:
                    metadata[key] = value  # Keep as string if conversion fails
        return metadata


    def extract_text_features(self, text):
        tokens = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output = roberta_model(**tokens)
        return output.hidden_states[-2][:,0,:]  # Shape: [1, 768]

    def extract_audio_features(self, audio_path):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load text
        with open(entry["text"], "r") as f:
            text = f.read().strip()
        text_features = self.extract_text_features(text)

        # Load audio
        audio_features = self.extract_audio_features(entry["audio"])

        # Extract metadata features
        meta_keys = [
            "Hours streamed", "Average viewers", "Peak viewers",
            "Hours watched", "Followers gained", "Followers / hour", "Followers"
        ]
        # metadata_features = torch.tensor([entry["metadata"].get(k, 0) for k in meta_keys], dtype=torch.float32)
        followers = entry["metadata"]["Followers"]
        return text_features.squeeze(0), audio_features.squeeze(0), followers

# Define the MultiModal MLP with Metadata
class MultiModalMLP(nn.Module):
    def __init__(self, text_dim, audio_dim, meta_dim, hidden_dim, output_dim):
        super(MultiModalMLP, self).__init__()
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
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, text_input, audio_input):
        text_features = self.text_mlp(text_input)
        audio_features = self.audio_mlp(audio_input)
        # meta_features = self.meta_mlp()
        fused = torch.cat((text_features, audio_features), dim=1)
        output = self.fusion_mlp(fused)
        return output

# Training function
def train_model(model, dataloader, epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for text_features, audio_features, metadata_features in dataloader:
            text_features, audio_features, metadata_features = (
                text_features.to(device), 
                audio_features.to(device), 
                metadata_features.to(device)
            )

            # Use log-scaled followers directly as the target
            target = torch.Tensor(metadata_features).to(device)
            optimizer.zero_grad()
            outputs = model(text_features, audio_features)
            loss = criterion(outputs, target)  # Loss in log-scale

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")



# Initialize dataset and dataloader
dataset = StreamerDataset("data/")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize and train the model
text_dim = 768  # RoBERTa embedding size
audio_dim = 768  # Wav2Vec2 embedding size
meta_dim = 7     # 7 metadata fields
hidden_dim = 128
output_dim = 1  # Regression (Follower count)

model = MultiModalMLP(text_dim, audio_dim, meta_dim, hidden_dim, output_dim)
train_model(model, dataloader, epochs=10, lr=1e-4)
