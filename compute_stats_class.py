import os
import numpy as np
import h5py
from classification import get_dict
from torch.utils.data import Dataset
import torch


class StreamerDatasetRaw(Dataset):
    def __init__(self, root_dir, label_dict, mode="both"):
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
        np.save(f"norm_params/{mod}_class_mean.npy", mean)
        np.save(f"norm_params/{mod}_class_std.npy", std)
        print(f"{mod.upper()} mean/std saved: shape = {mean.shape}")


if __name__ == "__main__":
    mode = "both"
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

    print(len(streamers))

    label_dict = get_dict(streamers)

    print(len(label_dict))
    dataset = StreamerDatasetRaw(root_dir="processed", label_dict=label_dict, mode=mode)
    compute_and_save_stats(dataset, mode, "")
