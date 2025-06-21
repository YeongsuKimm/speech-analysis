import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from classification import df
from torch.utils.data import Dataset, DataLoader, Subset

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Streamer Dataset (without metadata)
class StreamerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)

        self.data = []
        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            metadata_path = os.path.join(streamer_path, "metadata.pt")
            # Find all feature files
            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_")])
            
            metadata = torch.load(metadata_path)
            # print(metadata[2])
            
            for text_file, audio_file in zip(text_files, audio_files):
                text_path = os.path.join(streamer_path, text_file)
                audio_path = os.path.join(streamer_path, audio_file)
                self.data.append((text_path, audio_path, metadata[2]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_path, audio_path, metadata = self.data[idx]
        
        # Load precomputed features
        text_features = torch.load(text_path)  # Shape: (768,)
        audio_features = torch.load(audio_path)  # Shape: (768,)

        metadata = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0)

        return text_features.squeeze(0), audio_features.squeeze(0), metadata

# Define the MultiModal MLP (without metadata)
class MultiModalMLP(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, output_dim):
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
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, text_input, audio_input):
        text_features = self.text_mlp(text_input)  # (batch_size, hidden_dim)
        audio_features = self.audio_mlp(audio_input)  # (batch_size, hidden_dim)

        fused = torch.cat((text_features, audio_features), dim=1)  # (batch_size, hidden_dim * 2)
        output = self.fusion_mlp(fused)
        return output














# Training function
# def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device="cuda"):
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = nn.MSELoss()

#     for epoch in range(epochs):
#         total_train_loss = 0
#         total_val_loss = 0
        
#         # Training phase
#         model.train()
#         for text_features, audio_features, labels in train_loader:
#             text_features, audio_features, labels = (
#                 text_features.to(device),
#                 audio_features.to(device),
#                 labels.to(device),
#             )

#             optimizer.zero_grad()
#             outputs = model(text_features, audio_features)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)

#         # Validation phase
#         model.eval()
#         with torch.no_grad():
#             for text_features, audio_features, labels in val_loader:
#                 text_features, audio_features, labels = (
#                     text_features.to(device),
#                     audio_features.to(device),
#                     labels.to(device),
#                 )

#                 outputs = model(text_features, audio_features)
#                 loss = criterion(outputs, labels)
#                 total_val_loss += loss.item()

#         avg_val_loss = total_val_loss / len(val_loader)

#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


# from sklearn.model_selection import KFold
# from sklearn.ensemble import VotingRegressor

# if __name__ == "__main__":    
#     # Define K-fold cross-validation (k=4)
#     k_folds = 4
#     dataset_size = 163  # Assuming 100 total streamers
#     batch_size = 16
#     epochs = 20
#     lr = 1e-2

#     # Ensure directory exists
#     os.makedirs("models", exist_ok=True)

#     # Initialize dataset
#     dataset = StreamerDataset("processed/")
#     # Get list of streamers
#     streamer_folders = sorted(os.listdir("processed/"))
    
#     # Define model parameters
#     text_dim = 768  # RoBERTa embedding size
#     audio_dim = 768  # Wav2Vec2 embedding size
#     hidden_dim = 128
#     output_dim = 1  # Regression task (Follower count)
    
#     # K-Fold split by streamer
#     models = []
#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#     for fold, (train_idx, val_idx) in enumerate(kf.split(streamer_folders)):
#         train_streamers = [streamer_folders[i] for i in train_idx]
#         val_streamers = [streamer_folders[i] for i in val_idx]

#         # Create train and validation subsets using train_idx and val_idx for filtering based on streamer
#         train_indices = [i for i, (text_path, audio_path, metadata) in enumerate(dataset.data) if any(s in text_path for s in train_streamers)]
#         val_indices = [i for i, (text_path, audio_path, metadata) in enumerate(dataset.data) if any(s in text_path for s in val_streamers)]

#         train_subset = Subset(dataset, train_indices)
#         val_subset = Subset(dataset, val_indices)

#         print(f"\nFold {fold + 1}/{k_folds}")

#         # Create dataloaders
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#         # Initialize model for this fold
#         model = MultiModalMLP(text_dim, audio_dim, hidden_dim, output_dim)

#         # Train the model
#         train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

#         # Save the trained model for this fold
#         model_path = f"models/streamer_reg_model_fold{fold + 1}.pth"
#         torch.save(model.state_dict(), model_path)
#         print(f"Model for Fold {fold + 1} saved to {model_path}")
#         models.append(model)

#     # Ensemble Learning
#     print("\nEnsemble learning with Voting Regressor...")
#     # Assuming models are fitted and ready to be used for ensemble
#     ensemble_model = VotingRegressor(estimators=[(f"model_{i + 1}", models[i]) for i in range(k_folds)])

#     # Inference on validation set using ensemble
#     predictions = []
#     for fold, (train_idx, val_idx) in enumerate(kf.split(streamer_folders)):
#         train_streamers = [streamer_folders[i] for i in train_idx]
#         val_streamers = [streamer_folders[i] for i in val_idx]

#         # Filter dataset based on selected streamers
#         val_indices = [i for i, (text_path, audio_path, metadata) in enumerate(dataset.data) if any(s in text_path for s in val_streamers)]
#         val_subset = Subset(dataset, val_indices)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#         # Collect model predictions for this fold
#         fold_predictions = []
#         for batch in val_loader:
#             text_data, audio_data, _ = batch
#             with torch.no_grad():
#                 # Make predictions from all models in the ensemble
#                 fold_predictions.append(ensemble_model.predict([model(text_data, audio_data).cpu().numpy() for model in models]))

#         # Combine fold predictions (average out predictions)
#         predictions.append(np.mean(fold_predictions, axis=0))

#     # Save the ensemble model
#     ensemble_model_path = "models/ensemble_model.pth"
#     torch.save(ensemble_model, ensemble_model_path)
#     print(f"Ensemble model saved to {ensemble_model_path}")











#no K-Fold Cross-Validation 
def train_model(model, train_loader, epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        
        # Training phase
        model.train()
        for text_features, audio_features, labels in train_loader:
            text_features, audio_features, labels = (
                text_features.to(device),
                audio_features.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(text_features, audio_features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

if __name__ == "__main__":
    dataset_size = 145  # Assuming 100 total streamers
    batch_size = 16
    epochs = 30
    lr = 1e-2

    # Ensure directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize dataset
    dataset = StreamerDataset("processed/")
    # Get list of streamers
    streamer_folders = sorted(os.listdir("processed/"))
    
    # Define model parameters
    text_dim = 768  # RoBERTa embedding size
    audio_dim = 768  # Wav2Vec2 embedding size
    hidden_dim = 128
    output_dim = 1  # Regression task (Follower count)
    
    train_idx = [i for i in range(1,dataset_size)]

    train_streamers = [streamer_folders[i] for i in train_idx]

    train_indices = [i for i, (text_path, audio_path, metadata) in enumerate(dataset.data) if any(s in text_path for s in train_streamers)]

    train_subset = Subset(dataset, train_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Initialize model for this fold
    model = MultiModalMLP(text_dim, audio_dim, hidden_dim, output_dim)

    # Train the model
    train_model(model, train_loader, epochs=epochs, lr=lr)

    # Save the trained model for this fold
    model_path = f"models/streamer_reg_model_t80.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
