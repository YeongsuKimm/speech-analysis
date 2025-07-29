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
        self.mode = mode
        self.data = []

        fold_1 = ['xFSN_Saber', 'Zoomaa', 'zackrawrr', 'TheGeekEntry', 'Thiefs', 'TinaKitten', 'starsmitten', 'supertf', 'Sykkuno', 'robcdee', 'RTGame', 'SovietWomble', 'pupsker', 'Quin69', 'shroud', 'omareloff', 'PirateSoftware', 'RanbooLive', 'miia', 'pashaBiceps', 'nl_Kripp', 'LotharHS', 'MOONMOON', 'NateHill', 'kyliebitkin', 'LVNDMARK', 'Ludwig', 'jordansisco_', 'kyootbot', 'lilypichu', 'iLumpE', 'jasontheween', 'Joe_Bartolozzi', 'Glorious_E', 'Gorgc', 'iiTzTimmy', 'DGthe99', 'filian', 'Flight23white', 'cjya', 'Elajjaz', 'DisguisedToast', 'BrownGotti', 'Caedrel', 'Castro_1021', 'BennyCentral', 'A_Seagull', 'BobRoss', 'ahmpy']
        fold_2 = ['Wicked', 'vedal987', 'yourragegaming', 'T90Official', 'thesketchreal', 'TimTheTatman', 'Sideshow', 'SMii7Y', 'Sweet_Anita', 'redspecter23', 'RDCgaming', 'Sommerset', 'Psychoghost', 'QuarterJade', 'ShahZaM', 'NyyBeats', 'Pikabooirl', 'Rainbow6', 'MataraKan', 'Northernlion', 'Ninja', 'LFToxy_val', 'Mendo', 'Nadeshot', 'KmartPoker', 'LuluLuvely', 'LTANorth', 'JayOddity', 'Kitboga', 'Kyedae', 'hypnoshark', 'itsSpoit', 'JackManifoldTV', 'Geef', 'GoldGlove', 'Hiko', 'DEFAC3D', 'ExtraEmily', 'Fanum', 'Casson', 'Dyrus', 'CohhCarnage', 'BreesKnees', 'BrookeAB', 'caseoh_', 'Beardageddon', 'Aztecross', 'benjyfishy', 'Adapt']
        fold_3 = ['Vombuz', 'Valkyrae', 'xQc', 'survivalistaoe2de', 'Thebausffs', 'TenZ', 'Shotz', 'SmallAnt', 'SwaggerSouls', 'RedOpz', 'Ray__C', 'sodapoppin', 'PENTA', 'Punz', 'scump', 'MurderCrumpet', 'peterpark', 'plaqueboymax', 'MaryMaybe', 'Nmplol', 'Nihachu', 'LAXHAWTHORN007', 'MeatyMarley', 'MrSavage', 'KingWoolz', 'Lord_Kebun', 'loltyler1', 'Jacque', 'Keeoh', 'KaiCenat', 'huncho', 'Insym', 'ironmouse', 'Fannsy', 'GernaderJake', 'HasanAbi', 'd0cc_tv', 'EsfandTV', 'Emiru', 'carmen', 'chocoTaco', 'cloakzy', 'BreaK', 'BobbyPoffGaming', 'CaptainSparklez', 'BarbarousKing', 'AuzioMF', 'BadBoyHalo', '39daph']        
        fold_4 = ['Trynet123', 'Trick2g', 'x2Twins', 'Sterdekie', 'Terroriser', 'tarik', 'Shapaz', 'sapnaplive', 'summit1g', 'Rallied', 'Ray', 'sneakylol', 'p4perback', 'PontiacMadeDDG', 'ScreaM', 'mollozhang', 'Pestily', 'Philza', 'MARI', 'Necros', 'Nightblue3', 'LanceMcDonald', 'Maximilian_DOOD', 'moistcr1tikal', 'KidShadoe', 'lilsimsie', 'Loeya', 'J4CKIECHAN', 'k3soju', 'Jynxzi', 'HollywoodBob', 'iddqd', 'ImperialHal__', 'Everretta', 'fuslie', 'Gosu', 'crazyjapanese', 'erobb221', 'Duke', 'capturesca', 'Chap', 'Clix', 'Blue_Squadron', 'Bigpuffer', 'broxh_', 'AxialMatt', 'AussieAntics', 'Aydan', 'aceu']
        test = ['tjnv', 'TobiasFate', 'Tubbo', 'Stealthygolem', 'Swiftor', 'SypherPK', 'ScrubNoob', 'runthefutmarket', 'stableronaldo', 'RachtaZ', 'Ranger', 'sinatraa', 'OniKanaVT', 'POACH', 'Scarra', 'MisoxShiru', 'PaymoneyWubby', 'ohnePixel', 'Mactics', 'Nadia', 'NickEh30', 'L3WG', 'MacieJay', 'Mizkif', 'Kerrty', 'Lacy', 'LIRIK', 'ixxdeee', 'JonSandman', 'JoshOG', 'Gnomonkey', 'Hungrybox', 'imaqtpie', 'Eros', 'fl0m', 'forsen', 'Couriway', 'Emongg', 'DrLupo', 'BruceGreene', 'CDawgVA', 'Chica', 'BikeMan', 'bateson87', 'boxbox', 'AmericanDad', 'aircool', 'AustinShow']

        self.streamers = fold_1.copy()
        self.streamers.extend(fold_2)
        self.streamers.extend(fold_3)
        self.streamers.extend(fold_4)
        # self.streamers = os.listdir(root_dir)
        print(len(self.streamers))

        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            metadata_path = os.path.join(streamer_path, "metadata.h5")
            if not os.path.exists(metadata_path):
                continue
            
            with h5py.File(metadata_path, "r") as f:
                if "tensor" in f:
                    target = f["tensor"][2]
                    if target == 0.0:
                        print(streamer)
                        with open("todo.txt", "a") as f:
                            f.write(streamer + "\n")
                    print(target)  # Load scalar or array
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
        np.save(os.path.join(save_dir, "text_reg_mean.npy"), text_mean)
        np.save(os.path.join(save_dir, "text_reg_std.npy"), text_std)
        print("Saved text mean and std.")
    if audio_feats:
        audio_feats = np.stack(audio_feats)
        audio_mean = audio_feats.mean(axis=0)
        audio_std = audio_feats.std(axis=0) + 1e-8
        np.save(os.path.join(save_dir, "audio_reg_mean.npy"), audio_mean)
        np.save(os.path.join(save_dir, "audio_reg_std.npy"), audio_std)
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
    np.save(os.path.join(save_dir, "label_mean.npy"), label_mean.numpy())
    np.save(os.path.join(save_dir, "label_std.npy"), label_std.numpy())

if __name__ == "__main__":
    MODE = "both"
    dataset = StreamerDatasetRaw("processed/", mode=MODE)
    compute_and_save_stats(dataset, MODE)
    compute_and_save_label_stats(dataset)
