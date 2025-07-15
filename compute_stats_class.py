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


def compute_and_save_stats(dataset, mode, fold):
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
        np.save(f"norm_params/fold{fold}_{mod}_class_mean.npy", mean)
        np.save(f"norm_params/fold{fold}_{mod}_class_std.npy", std)
        print(f"{mod.upper()} mean/std saved: shape = {mean.shape}")


if __name__ == "__main__":
    mode = "both"
    root_dir = "processed"
    fold_1 = ['ahmpy', 'AmericanDad', 'AxialMatt', 'BarbarousKing', 'Beardageddon', 'BennyCentral', 'BikeMan', 'Blue_Squadron', 'BreaK', 'BreesKnees', 'BrownGotti', 'BruceGreene', 'capturesca', 'carmen', 'Casson', 'cjya', 'Couriway', 'crazyjapanese', 'd0cc_tv', 'DEFAC3D', 'DGthe99', 'Eros', 'Everretta', 'Fannsy', 'Geef', 'Glorious_E', 'Gnomonkey', 'HollywoodBob', 'huncho', 'hypnoshark', 'iLumpE', 'ixxdeee', 'J4CKIECHAN', 'Jacque', 'JayOddity', 'jordansisco_', 'Kerrty', 'KidShadoe', 'KingWoolz', 'KmartPoker', 'kyliebitkin', 'L3WG', 'LanceMcDonald', 'LAXHAWTHORN007', 'LFToxy_val', 'LotharHS', 'Mactics', 'MARI', 'MaryMaybe']
    fold_2 = ['MataraKan', 'miia', 'MisoxShiru', 'mollozhang', 'MurderCrumpet', 'NyyBeats', 'omareloff', 'OniKanaVT', 'p4perback', 'PENTA', 'Psychoghost', 'pupsker', 'RachtaZ', 'Rallied', 'RedOpz', 'redspecter23', 'robcdee', 'ScrubNoob', 'Shapaz', 'Shotz', 'Sideshow', 'starsmitten', 'Stealthygolem', 'Sterdekie', 'survivalistaoe2de', 'T90Official', 'TheGeekEntry', 'tjnv', 'Trynet123', 'Vombuz', 'Wicked', 'xFSN_Saber', '39daph', 'Adapt', 'aircool', 'AussieAntics', 'AuzioMF', 'Aztecross', 'A_Seagull', 'bateson87', 'Bigpuffer', 'BobbyPoffGaming', 'BrookeAB', 'Caedrel', 'CDawgVA', 'Chap', 'chocoTaco', 'Dyrus', 'Elajjaz']
    fold_3 = ['Emongg', 'erobb221', 'EsfandTV', 'ExtraEmily', 'filian', 'fl0m', 'fuslie', 'GernaderJake', 'GoldGlove', 'Gorgc', 'Hungrybox', 'iddqd', 'Insym', 'itsSpoit', 'jasontheween', 'JonSandman', 'k3soju', 'Keeoh', 'Kitboga', 'kyootbot', 'Lacy', 'lilsimsie', 'Lord_Kebun', 'LuluLuvely', 'LVNDMARK', 'MacieJay', 'Maximilian_DOOD', 'MeatyMarley', 'Mendo', 'MOONMOON', 'Nadia', 'Necros', 'Nmplol', 'Northernlion', 'pashaBiceps', 'PaymoneyWubby', 'Pestily', 'peterpark', 'Pikabooirl', 'PirateSoftware', 'POACH', 'PontiacMadeDDG', 'Punz', 'QuarterJade', 'Quin69', 'Ranger', 'Ray', 'Ray__C', 'RDCgaming']
    fold_4 = ['RTGame', 'runthefutmarket', 'sapnaplive', 'SmallAnt', 'SMii7Y', 'supertf', 'Swiftor', 'Terroriser', 'Thebausffs', 'thesketchreal', 'Thiefs', 'TobiasFate', 'Trick2g', 'Valkyrae', 'vedal987', 'Zoomaa', 'aceu', 'AustinShow', 'Aydan', 'BadBoyHalo', 'benjyfishy', 'BobRoss', 'boxbox', 'broxh_', 'CaptainSparklez', 'caseoh_', 'Castro_1021', 'Chica', 'Clix', 'cloakzy', 'CohhCarnage', 'DisguisedToast', 'DrLupo', 'Duke', 'Emiru', 'Fanum', 'Flight23white', 'forsen', 'Gosu', 'HasanAbi', 'Hiko', 'iiTzTimmy', 'imaqtpie', 'ImperialHal__', 'ironmouse', 'JackManifoldTV', 'Joe_Bartolozzi', 'JoshOG', 'Jynxzi']
    # label_dict = get_dict([item for item in os.listdir(root_dir) if item not in fold_4])
    label_dict = get_dict(os.listdir(root_dir))
    print(len(label_dict))
    dataset = StreamerDatasetRaw(root_dir=root_dir, label_dict=label_dict, mode=mode)
    compute_and_save_stats(dataset, mode, "")
