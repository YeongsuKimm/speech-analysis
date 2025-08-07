import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from classification import get_dict
import h5py
import numpy as np
from sklearn.model_selection import KFold


# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mode selection: "text", "audio", or "both"
MODE = "text"

# Define the Streamer Dataset (supports all modes)
class StreamerDataset(Dataset):
    def __init__(self, root_dir, label_dict, mode="both", normalize=True):
        self.root_dir = root_dir
        # self.streamers = os.listdir(root_dir)

        fold_1 = ['xFSN_Saber', 'Zoomaa', 'zackrawrr', 'TheGeekEntry', 'Thiefs', 'TinaKitten', 'starsmitten', 'supertf', 'Sykkuno', 'robcdee', 'RTGame', 'SovietWomble', 'pupsker', 'Quin69', 'shroud', 'omareloff', 'PirateSoftware', 'RanbooLive', 'miia', 'pashaBiceps', 'nl_Kripp', 'LotharHS', 'MOONMOON', 'NateHill', 'kyliebitkin', 'LVNDMARK', 'Ludwig', 'jordansisco_', 'kyootbot', 'lilypichu', 'iLumpE', 'jasontheween', 'Joe_Bartolozzi', 'Glorious_E', 'Gorgc', 'iiTzTimmy', 'DGthe99', 'filian', 'Flight23white', 'cjya', 'Elajjaz', 'DisguisedToast', 'BrownGotti', 'Caedrel', 'Castro_1021', 'BennyCentral', 'A_Seagull', 'BobRoss', 'ahmpy']
        fold_2 = ['Wicked', 'vedal987', 'yourragegaming', 'T90Official', 'thesketchreal', 'TimTheTatman', 'Sideshow', 'SMii7Y', 'Sweet_Anita', 'redspecter23', 'RDCgaming', 'Sommerset', 'Psychoghost', 'QuarterJade', 'ShahZaM', 'NyyBeats', 'Pikabooirl', 'Rainbow6', 'MataraKan', 'Northernlion', 'Ninja', 'LFToxy_val', 'Mendo', 'Nadeshot', 'KmartPoker', 'LuluLuvely', 'LTANorth', 'JayOddity', 'Kitboga', 'Kyedae', 'hypnoshark', 'itsSpoit', 'JackManifoldTV', 'Geef', 'GoldGlove', 'Hiko', 'DEFAC3D', 'ExtraEmily', 'Fanum', 'Casson', 'Dyrus', 'CohhCarnage', 'BreesKnees', 'BrookeAB', 'caseoh_', 'Beardageddon', 'Aztecross', 'benjyfishy', 'Adapt']
        fold_3 = ['Vombuz', 'Valkyrae', 'xQc', 'survivalistaoe2de', 'Thebausffs', 'TenZ', 'Shotz', 'SmallAnt', 'SwaggerSouls', 'RedOpz', 'Ray__C', 'sodapoppin', 'PENTA', 'Punz', 'scump', 'MurderCrumpet', 'peterpark', 'plaqueboymax', 'MaryMaybe', 'Nmplol', 'Nihachu', 'LAXHAWTHORN007', 'MeatyMarley', 'MrSavage', 'KingWoolz', 'Lord_Kebun', 'loltyler1', 'Jacque', 'Keeoh', 'KaiCenat', 'huncho', 'Insym', 'ironmouse', 'Fannsy', 'GernaderJake', 'HasanAbi', 'd0cc_tv', 'EsfandTV', 'Emiru', 'carmen', 'chocoTaco', 'cloakzy', 'BreaK', 'BobbyPoffGaming', 'CaptainSparklez', 'BarbarousKing', 'AuzioMF', 'BadBoyHalo', '39daph']        
        fold_4 = ['Trynet123', 'Trick2g', 'x2Twins', 'Sterdekie', 'Terroriser', 'tarik', 'Shapaz', 'sapnaplive', 'summit1g', 'Rallied', 'Ray', 'sneakylol', 'p4perback', 'PontiacMadeDDG', 'ScreaM', 'mollozhang', 'Pestily', 'Philza', 'MARI', 'Necros', 'Nightblue3', 'LanceMcDonald', 'Maximilian_DOOD', 'moistcr1tikal', 'KidShadoe', 'lilsimsie', 'Loeya', 'J4CKIECHAN', 'k3soju', 'Jynxzi', 'HollywoodBob', 'iddqd', 'ImperialHal__', 'Everretta', 'fuslie', 'Gosu', 'crazyjapanese', 'erobb221', 'Duke', 'capturesca', 'Chap', 'Clix', 'Blue_Squadron', 'Bigpuffer', 'broxh_', 'AxialMatt', 'AussieAntics', 'Aydan', 'aceu']
        test = ['tjnv', 'TobiasFate', 'Tubbo', 'Stealthygolem', 'Swiftor', 'SypherPK', 'ScrubNoob', 'runthefutmarket', 'stableronaldo', 'RachtaZ', 'Ranger', 'sinatraa', 'OniKanaVT', 'POACH', 'Scarra', 'MisoxShiru', 'PaymoneyWubby', 'ohnePixel', 'Mactics', 'Nadia', 'NickEh30', 'L3WG', 'MacieJay', 'Mizkif', 'Kerrty', 'Lacy', 'LIRIK', 'ixxdeee', 'JonSandman', 'JoshOG', 'Gnomonkey', 'Hungrybox', 'imaqtpie', 'Eros', 'fl0m', 'forsen', 'Couriway', 'Emongg', 'DrLupo', 'BruceGreene', 'CDawgVA', 'Chica', 'BikeMan', 'bateson87', 'boxbox', 'AmericanDad', 'aircool', 'AustinShow']

        self.streamers = fold_1.copy()
        self.streamers.extend(fold_2)
        self.streamers.extend(fold_3)
        self.streamers.extend(fold_4)

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
                self.mean_text = torch.tensor(np.load(f"norm_params/text_class_mean.npy"), dtype=torch.float32)
                self.std_text = torch.tensor(np.load(f"norm_params/text_class_std.npy"), dtype=torch.float32)
            if mode in ["audio", "both"]:
                self.mean_audio = torch.tensor(np.load(f"norm_params/audio_class_mean.npy"), dtype=torch.float32)
                self.std_audio = torch.tensor(np.load(f"norm_params/audio_class_std.npy"), dtype=torch.float32)

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
            # print(text_path)
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
                # print("p")
                # print(preds)
                # print(labels)
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
    NUM_EPOCHS = 25
    LR = 1e-3

    fold_1 = ['xFSN_Saber', 'Zoomaa', 'zackrawrr', 'TheGeekEntry', 'Thiefs', 'TinaKitten', 'starsmitten', 'supertf', 'Sykkuno', 'robcdee', 'RTGame', 'SovietWomble', 'pupsker', 'Quin69', 'shroud', 'omareloff', 'PirateSoftware', 'RanbooLive', 'miia', 'pashaBiceps', 'nl_Kripp', 'LotharHS', 'MOONMOON', 'NateHill', 'kyliebitkin', 'LVNDMARK', 'Ludwig', 'jordansisco_', 'kyootbot', 'lilypichu', 'iLumpE', 'jasontheween', 'Joe_Bartolozzi', 'Glorious_E', 'Gorgc', 'iiTzTimmy', 'DGthe99', 'filian', 'Flight23white', 'cjya', 'Elajjaz', 'DisguisedToast', 'BrownGotti', 'Caedrel', 'Castro_1021', 'BennyCentral', 'A_Seagull', 'BobRoss', 'ahmpy']
    fold_2 = ['Wicked', 'vedal987', 'yourragegaming', 'T90Official', 'thesketchreal', 'TimTheTatman', 'Sideshow', 'SMii7Y', 'Sweet_Anita', 'redspecter23', 'RDCgaming', 'Sommerset', 'Psychoghost', 'QuarterJade', 'ShahZaM', 'NyyBeats', 'Pikabooirl', 'Rainbow6', 'MataraKan', 'Northernlion', 'Ninja', 'LFToxy_val', 'Mendo', 'Nadeshot', 'KmartPoker', 'LuluLuvely', 'LTANorth', 'JayOddity', 'Kitboga', 'Kyedae', 'hypnoshark', 'itsSpoit', 'JackManifoldTV', 'Geef', 'GoldGlove', 'Hiko', 'DEFAC3D', 'ExtraEmily', 'Fanum', 'Casson', 'Dyrus', 'CohhCarnage', 'BreesKnees', 'BrookeAB', 'caseoh_', 'Beardageddon', 'Aztecross', 'benjyfishy', 'Adapt']
    fold_3 = ['Vombuz', 'Valkyrae', 'xQc', 'survivalistaoe2de', 'Thebausffs', 'TenZ', 'Shotz', 'SmallAnt', 'SwaggerSouls', 'RedOpz', 'Ray__C', 'sodapoppin', 'PENTA', 'Punz', 'scump', 'MurderCrumpet', 'peterpark', 'plaqueboymax', 'MaryMaybe', 'Nmplol', 'Nihachu', 'LAXHAWTHORN007', 'MeatyMarley', 'MrSavage', 'KingWoolz', 'Lord_Kebun', 'loltyler1', 'Jacque', 'Keeoh', 'KaiCenat', 'huncho', 'Insym', 'ironmouse', 'Fannsy', 'GernaderJake', 'HasanAbi', 'd0cc_tv', 'EsfandTV', 'Emiru', 'carmen', 'chocoTaco', 'cloakzy', 'BreaK', 'BobbyPoffGaming', 'CaptainSparklez', 'BarbarousKing', 'AuzioMF', 'BadBoyHalo', '39daph']        
    fold_4 = ['Trynet123', 'Trick2g', 'x2Twins', 'Sterdekie', 'Terroriser', 'tarik', 'Shapaz', 'sapnaplive', 'summit1g', 'Rallied', 'Ray', 'sneakylol', 'p4perback', 'PontiacMadeDDG', 'ScreaM', 'mollozhang', 'Pestily', 'Philza', 'MARI', 'Necros', 'Nightblue3', 'LanceMcDonald', 'Maximilian_DOOD', 'moistcr1tikal', 'KidShadoe', 'lilsimsie', 'Loeya', 'J4CKIECHAN', 'k3soju', 'Jynxzi', 'HollywoodBob', 'iddqd', 'ImperialHal__', 'Everretta', 'fuslie', 'Gosu', 'crazyjapanese', 'erobb221', 'Duke', 'capturesca', 'Chap', 'Clix', 'Blue_Squadron', 'Bigpuffer', 'broxh_', 'AxialMatt', 'AussieAntics', 'Aydan', 'aceu']
    test = ['tjnv', 'TobiasFate', 'Tubbo', 'Stealthygolem', 'Swiftor', 'SypherPK', 'ScrubNoob', 'runthefutmarket', 'stableronaldo', 'RachtaZ', 'Ranger', 'sinatraa', 'OniKanaVT', 'POACH', 'Scarra', 'MisoxShiru', 'PaymoneyWubby', 'ohnePixel', 'Mactics', 'Nadia', 'NickEh30', 'L3WG', 'MacieJay', 'Mizkif', 'Kerrty', 'Lacy', 'LIRIK', 'ixxdeee', 'JonSandman', 'JoshOG', 'Gnomonkey', 'Hungrybox', 'imaqtpie', 'Eros', 'fl0m', 'forsen', 'Couriway', 'Emongg', 'DrLupo', 'BruceGreene', 'CDawgVA', 'Chica', 'BikeMan', 'bateson87', 'boxbox', 'AmericanDad', 'aircool', 'AustinShow']
    folds = [fold_1, fold_2, fold_3, fold_4]

    combined = fold_1.copy()
    combined.extend(fold_2)
    combined.extend(fold_3)
    combined.extend(fold_4)
    combined.extend(test)
    streamer_list=combined

    label_dict = get_dict(combined)
    dataset = StreamerDataset(root_dir="processed", label_dict=label_dict, mode=MODE, normalize=True)

    # Make a mapping from streamer name to sample indices
    streamer_to_indices = {}
    for idx, item in enumerate(dataset.data):
        if MODE == "both":
            path = item[0]  # text path
        else:
            path = item[0]  # text or audio path
        streamer_name = os.path.basename(os.path.dirname(path))
        if streamer_name not in streamer_to_indices:
            streamer_to_indices[streamer_name] = []
        streamer_to_indices[streamer_name].append(idx)

    accuracies = []
    K = 4

    for fold in range(K):
        val_streamers = set(folds[fold])
        train_streamers = set.union(*[set(folds[i]) for i in range(K) if i != fold])

        val_indices = [idx for s in val_streamers for idx in streamer_to_indices.get(s, [])]
        train_indices = [idx for s in train_streamers for idx in streamer_to_indices.get(s, [])]

        print(f"\n--- Fold {fold + 1} ---")
        print(f"Train streamers: {len(train_streamers)}, Validation streamers: {len(val_streamers)}")
        print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        if MODE == "text":
            model = TextOnlyMLP(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        elif MODE == "audio":
            model = AudioOnlyMLP(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        else:
            model = MultiModalMLP(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=NUM_EPOCHS)

        # Evaluate on val set
        model.eval()
        correct = 0
        total = 0
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

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        accuracies.append(acc)
        print(f"Fold {fold + 1} Final Accuracy: {acc:.2f}%")
        torch.save(model.state_dict(), f"models/{MODE}_class_model_fold_{fold + 1}.pt")

    print(f"\n=== Average Accuracy across {K} folds: {np.mean(accuracies):.2f}% ===")