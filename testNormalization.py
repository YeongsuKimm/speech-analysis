import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from classification import get_dict
from torch.utils.data import Dataset, DataLoader, Subset

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Streamer Dataset (without metadata)
class StreamerDataset(Dataset):
    def __init__(self, root_dir, label_dict, normalize=True):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.label_dict = label_dict
        self.normalize = normalize

        self.data = []
        for streamer in self.streamers:
            if streamer not in self.label_dict:
                continue

            streamer_path = os.path.join(root_dir, streamer)
            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_")])

            label = self.label_dict[streamer]

            for text_file, audio_file in zip(text_files, audio_files):
                text_path = os.path.join(streamer_path, text_file)
                audio_path = os.path.join(streamer_path, audio_file)
                self.data.append((text_path, audio_path, label))

        if self.normalize:
            self.compute_normalization_stats()
        print('test')
        
    def compute_normalization_stats(self):
        all_text = []
        all_audio = []

        for text_path, audio_path, _ in self.data:
            # Force loading on CPU
            text_feat = torch.load(text_path, map_location='cpu').squeeze(0)
            audio_feat = torch.load(audio_path, map_location='cpu').squeeze(0)
            print(text_feat.device)
            all_text.append(text_feat)
            all_audio.append(audio_feat)

        all_text = torch.stack(all_text)
        all_audio = torch.stack(all_audio)

        self.text_mean = all_text.mean(dim=0)
        self.text_std = all_text.std(dim=0) + 1e-6  # to prevent division by zero
        self.audio_mean = all_audio.mean(dim=0)
        self.audio_std = all_audio.std(dim=0) + 1e-6


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        text_path, audio_path, label = self.data[idx]
        text_features = torch.load(text_path).squeeze(0)
        audio_features = torch.load(audio_path).squeeze(0)

        if self.normalize:
            text_features = (text_features - self.text_mean) / self.text_std
            audio_features = (audio_features - self.audio_mean) / self.audio_std

        label = torch.tensor(label, dtype=torch.long)
        return text_features, audio_features, label

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
            nn.Linear(hidden_dim, output_dim)  # Adjusted for regression/classification
        )
        # Add batchnorm and dropout

    def forward(self, text_input, audio_input):
        text_features = self.text_mlp(text_input)  # (batch_size, hidden_dim)
        audio_features = self.audio_mlp(audio_input)  # (batch_size, hidden_dim)

        fused = torch.cat((text_features, audio_features), dim=1)  # (batch_size, hidden_dim * 2)
        output = self.fusion_mlp(fused)
        return output

def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    model.to(DEVICE)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for text, audio, labels in dataloader:
            text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(text, audio)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    streamer_list = []
    for streamer in os.listdir("processed"):
        streamer_list.append(streamer)
    label_dict = get_dict(streamer_list)
    dataset = StreamerDataset(root_dir="processed", label_dict=label_dict, normalize=True)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MultiModalMLP(text_dim=768, audio_dim=768, hidden_dim=64, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_model(model, dataloader, optimizer, criterion, num_epochs=10)
    model_path = f"models/streamer_nclass_model_t70.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# from sklearn.model_selection import KFold
# from sklearn.ensemble import VotingRegressor
# if __name__ == "__main__":
#     def train_one_epoch(model, dataloader, optimizer, criterion):
#         model.train()
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         for text, audio, labels in dataloader:
#             text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(text, audio)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#         acc = correct / total * 100
#         return total_loss, acc

#     def validate(model, dataloader, criterion):
#         model.eval()
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for text, audio, labels in dataloader:
#                 text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

#                 outputs = model(text, audio)
#                 loss = criterion(outputs, labels)

#                 total_loss += loss.item()
#                 preds = torch.argmax(outputs, dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         acc = correct / total * 100
#         return total_loss, acc

#     def kfold_cross_validation(model, dataset, k=4, num_epochs=10, lr=1e-2, save_dir="models"):
#         kfold = KFold(n_splits=k, shuffle=True)
#         fold_results = []

#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
#             print(f"\n--- Training fold {fold + 1}/{k} ---")

#             train_subset = Subset(dataset, train_idx)
#             val_subset = Subset(dataset, val_idx)

#             train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True)
#             val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False)

#             model_fold = MultiModalMLP(text_dim=768, audio_dim=768, hidden_dim=128, output_dim=3).to(DEVICE)
#             optimizer = optim.Adam(model_fold.parameters(), lr=lr)
#             criterion = nn.CrossEntropyLoss()

#             for epoch in range(1, num_epochs + 1):
#                 train_loss, train_acc = train_one_epoch(model_fold, train_dataloader, optimizer, criterion)
#                 val_loss, val_acc = validate(model_fold, val_dataloader, criterion)

#                 print(f"Epoch {epoch}/{num_epochs} | "
#                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
#                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

#             # Save model after last epoch of this fold
#             model_save_path = os.path.join(save_dir, f"streamer_class_model_fold{fold + 1}.pth")
#             torch.save(model_fold.state_dict(), model_save_path)
#             print(f"Model for fold {fold + 1} saved to {model_save_path}")

#             # Save final epoch metrics for this fold
#             fold_results.append((train_loss, train_acc, val_loss, val_acc))

#         # Print summary
#         avg_train_loss = np.mean([result[0] for result in fold_results])
#         avg_train_acc = np.mean([result[1] for result in fold_results])
#         avg_val_loss = np.mean([result[2] for result in fold_results])
#         avg_val_acc = np.mean([result[3] for result in fold_results])

#         print("\n=== Overall Cross-Validation Results ===")
#         print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_acc:.2f}%")
#         print(f"Avg Validation Loss: {avg_val_loss:.4f}, Avg Validation Accuracy: {avg_val_acc:.2f}%")

#     # Example usage
#     streamer_list = os.listdir("processed")
#     label_dict = get_dict(streamer_list)
#     dataset = StreamerDataset(root_dir="processed", label_dict=label_dict)

#     model = MultiModalMLP(text_dim=768, audio_dim=768, hidden_dim=128, output_dim=3)

#     # Perform K-Fold Cross-Validation with model saving
#     kfold_cross_validation(model, dataset, k=4, num_epochs=30, lr=1e-2, save_dir="models")
