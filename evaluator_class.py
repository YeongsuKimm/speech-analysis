import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from separateClass import MultiModalMLP, TextOnlyMLP, AudioOnlyMLP 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "both"  # change as needed: "text", "audio", or "both"
BATCH_SIZE = 16

# Load your label dictionary function or data
from classification import get_dict

# Normalization flags

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
                # print(labels)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                # print(preds)

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



if __name__ == "__main__":
    NORMALIZE = True
    fold_1 = ['xFSN_Saber', 'Zoomaa', 'zackrawrr', 'TheGeekEntry', 'Thiefs', 'TinaKitten', 'starsmitten', 'supertf', 'Sykkuno', 'robcdee', 'RTGame', 'SovietWomble', 'pupsker', 'Quin69', 'shroud', 'omareloff', 'PirateSoftware', 'RanbooLive', 'miia', 'pashaBiceps', 'nl_Kripp', 'LotharHS', 'MOONMOON', 'NateHill', 'kyliebitkin', 'LVNDMARK', 'Ludwig', 'jordansisco_', 'kyootbot', 'lilypichu', 'iLumpE', 'jasontheween', 'Joe_Bartolozzi', 'Glorious_E', 'Gorgc', 'iiTzTimmy', 'DGthe99', 'filian', 'Flight23white', 'cjya', 'Elajjaz', 'DisguisedToast', 'BrownGotti', 'Caedrel', 'Castro_1021', 'BennyCentral', 'A_Seagull', 'BobRoss', 'ahmpy']
    fold_2 = ['Wicked', 'vedal987', 'yourragegaming', 'T90Official', 'thesketchreal', 'TimTheTatman', 'Sideshow', 'SMii7Y', 'Sweet_Anita', 'redspecter23', 'RDCgaming', 'Sommerset', 'Psychoghost', 'QuarterJade', 'ShahZaM', 'NyyBeats', 'Pikabooirl', 'Rainbow6', 'MataraKan', 'Northernlion', 'Ninja', 'LFToxy_val', 'Mendo', 'Nadeshot', 'KmartPoker', 'LuluLuvely', 'LTANorth', 'JayOddity', 'Kitboga', 'Kyedae', 'hypnoshark', 'itsSpoit', 'JackManifoldTV', 'Geef', 'GoldGlove', 'Hiko', 'DEFAC3D', 'ExtraEmily', 'Fanum', 'Casson', 'Dyrus', 'CohhCarnage', 'BreesKnees', 'BrookeAB', 'caseoh_', 'Beardageddon', 'Aztecross', 'benjyfishy', 'Adapt']
    fold_3 = ['Vombuz', 'Valkyrae', 'xQc', 'survivalistaoe2de', 'Thebausffs', 'TenZ', 'Shotz', 'SmallAnt', 'SwaggerSouls', 'RedOpz', 'Ray__C', 'sodapoppin', 'PENTA', 'Punz', 'scump', 'MurderCrumpet', 'peterpark', 'plaqueboymax', 'MaryMaybe', 'Nmplol', 'Nihachu', 'LAXHAWTHORN007', 'MeatyMarley', 'MrSavage', 'KingWoolz', 'Lord_Kebun', 'loltyler1', 'Jacque', 'Keeoh', 'KaiCenat', 'huncho', 'Insym', 'ironmouse', 'Fannsy', 'GernaderJake', 'HasanAbi', 'd0cc_tv', 'EsfandTV', 'Emiru', 'carmen', 'chocoTaco', 'cloakzy', 'BreaK', 'BobbyPoffGaming', 'CaptainSparklez', 'BarbarousKing', 'AuzioMF', 'BadBoyHalo', '39daph']        
    fold_4 = ['Trynet123', 'Trick2g', 'x2Twins', 'Sterdekie', 'Terroriser', 'tarik', 'Shapaz', 'sapnaplive', 'summit1g', 'Rallied', 'Ray', 'sneakylol', 'p4perback', 'PontiacMadeDDG', 'ScreaM', 'mollozhang', 'Pestily', 'Philza', 'MARI', 'Necros', 'Nightblue3', 'LanceMcDonald', 'Maximilian_DOOD', 'moistcr1tikal', 'KidShadoe', 'lilsimsie', 'Loeya', 'J4CKIECHAN', 'k3soju', 'Jynxzi', 'HollywoodBob', 'iddqd', 'ImperialHal__', 'Everretta', 'fuslie', 'Gosu', 'crazyjapanese', 'erobb221', 'Duke', 'capturesca', 'Chap', 'Clix', 'Blue_Squadron', 'Bigpuffer', 'broxh_', 'AxialMatt', 'AussieAntics', 'Aydan', 'aceu']
    test = ['tjnv', 'TobiasFate', 'Tubbo', 'Stealthygolem', 'Swiftor', 'SypherPK', 'ScrubNoob', 'runthefutmarket', 'stableronaldo', 'RachtaZ', 'Ranger', 'sinatraa', 'OniKanaVT', 'POACH', 'Scarra', 'MisoxShiru', 'PaymoneyWubby', 'ohnePixel', 'Mactics', 'Nadia', 'NickEh30', 'L3WG', 'MacieJay', 'Mizkif', 'Kerrty', 'Lacy', 'LIRIK', 'ixxdeee', 'JonSandman', 'JoshOG', 'Gnomonkey', 'Hungrybox', 'imaqtpie', 'Eros', 'fl0m', 'forsen', 'Couriway', 'Emongg', 'DrLupo', 'BruceGreene', 'CDawgVA', 'Chica', 'BikeMan', 'bateson87', 'boxbox', 'AmericanDad', 'aircool', 'AustinShow']

    streamers = fold_1.copy()
    streamers.extend(fold_2)
    streamers.extend(fold_3)
    streamers.extend(fold_4)
    streamers.extend(test)

    label_dict = get_dict(streamers)

    


    # models = ["models/audio_only_class_model_unnormalized_t70-3-wd-3.pth", "models/audio_only_class_model_unnormalized_t70-3-wd-7.pth", "models/streamer_class_model_unnormalized_t70-3-wd-3.pth", 
    #           "models/streamer_class_model_unnormalized_t70-3-wd-5.pth", "models/streamer_class_model_unnormalized_t70-3-wd-7.pth", "models/text_only_class_model_unnormalized_t70-3-wd-3.pth",
    #           "models/text_only_class_model_unnormalized_t70-3-wd-7.pth"]

    # models = ["models/audio_only_class_model_normalized_t70-3-wd-3.pth", "models/audio_only_class_model_normalized_t70-3-wd-7.pth", "models/text_only_class_model_normalized_t70-3-wd-3.pth",
    #           "models/text_only_class_model_normalized_t70-3-wd-7.pth", "models/streamer_class_model_normalized_t70-3-wd-3.pth", "models/streamer_class_model_normalized_t70-3-wd-5.pth",
    #           "models/streamer_class_model_normalized_t70-3-wd-7.pth"
    # ]

    models = ['models/text_class_model_fold_1.pt', 'models/text_class_model_fold_2.pt', 'models/text_class_model_fold_3.pt', 'models/text_class_model_fold_4.pt']
    
    TEXT_DIM = AUDIO_DIM = 768  # adjust if needed
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3

    for i in models:
        if "audio" in i:
            MODE = "audio"
        elif "text" in i:
            MODE = "text"
        else:
            MODE = "both"

        dataset = StreamerDataset(root_dir=".test", label_dict=label_dict, mode=MODE, normalize=NORMALIZE)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        if MODE == "text":
            model = TextOnlyMLP(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/text_only_class_model_normalized_t70-3.pth"
        elif MODE == "audio":
            model = AudioOnlyMLP(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/audio_only_class_model_normalized_t70-3.pth"
        else:
            model = MultiModalMLP(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/streamer_class_model_normalized_t70-3.pth"

        model_path = i
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        model = model

        criterion = nn.CrossEntropyLoss()

        evaluator = Evaluator(model, criterion, DEVICE)
        loss, accuracy = evaluator.evaluate(data_loader, mode=MODE)
        with open("results.txt", "a") as f:
            f.write(f"{model_path}\n")
            f.write(f"Evaluation Loss: {loss:.4f}\n")
            f.write(f"Evaluation Accuracy: {accuracy:.2f}%\n\n")
