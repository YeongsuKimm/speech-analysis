import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from separateReg import (  # Replace with your actual filename where classes are defined
    StreamerDataset,
    TextOnlyRegressor,
    AudioOnlyRegressor,
    MultiModalRegressor
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model, dataloader, mode="both", norm_dir=None, device="cuda"):
    model.eval()
    model.to(device)
    criterion = torch.nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if mode == "both":
                text, audio, target = batch
                text, audio, target = text.to(device), audio.to(device), target.to(device)
                outputs = model(text, audio).squeeze()
            else:
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs).squeeze()

            if norm_dir:
                label_mean = torch.tensor(np.load(f"{norm_dir}/label_mean.npy")).to(device)
                label_std = torch.tensor(np.load(f"{norm_dir}/label_std.npy")).to(device)
                outputs = outputs * label_std + label_mean
                target = target * label_std + label_mean

            loss = criterion(outputs, target.squeeze())
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(dataloader)
    print(f"Validation MSE Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    # models = ["models/audio_only_reg_model_normalized_t70-3-wd-3.pth", "models/audio_only_reg_model_normalized_t70-3-wd-7.pth", "models/streamer_reg_model_normalized_t70-3-wd-3.pth",
    #         "models/streamer_reg_model_normalized_t70-3-wd-5.pth", "models/streamer_reg_model_normalized_t70-3-wd-7.pth", "models/text_only_reg_model_normalized_t70-3-wd-3.pth",
    #         "models/text_only_reg_model_normalized_t70-3-wd-7.pth"]

    models = ['models/fold_reg_1_audio_model.pth', 'models/fold_reg_1_both_model.pth', 'models/fold_reg_1_text_model.pth', 'models/fold_reg_2_audio_model.pth', 'models/fold_reg_2_both_model.pth', 'models/fold_reg_2_text_model.pth', 'models/fold_reg_3_audio_model.pth', 'models/fold_reg_3_both_model.pth', 'models/fold_reg_3_text_model.pth', 'models/fold_reg_4_audio_model.pth', 'models/fold_reg_4_both_model.pth', 'models/fold_reg_4_text_model.pth']
    for i in models:
        model_path = i
        if "text" in i:
            MODE = "text"
        elif "audio" in i:
            MODE = "audio"
        else:
            MODE = "both"
        
        dataset = StreamerDataset(".test/", mode=MODE, norm_dir="norm_params")

        # You can split dataset or create val_loader like in your training script
        from torch.utils.data import Subset

        val_streamers = sorted(os.listdir(".test/"))

        val_indices = [i for i, entry in enumerate(dataset.data) if any(s in entry[0] for s in val_streamers)]
        val_dataset = Subset(dataset, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

        # --- Load model ---
        TEXT_DIM = AUDIO_DIM = 768
        HIDDEN_DIM = 128
        OUTPUT_DIM = 1

        if MODE == "text":
            model = TextOnlyRegressor(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/text_only_reg_model_normalized_t70-3.pth"
        elif MODE == "audio":
            model = AudioOnlyRegressor(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/audio_only_reg_model_normalized_t70-3.pth"
        else:
            model = MultiModalRegressor(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
            # model_path = "models/streamer_reg_model_normalized_t70-3.pth"

        
        print(model_path)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # --- Evaluate ---
        avg_loss = evaluate_model(model, val_loader, mode=MODE, norm_dir="norm_params", device=DEVICE)

        with open("results.txt", "a") as f:
            f.write(f"{model_path}\n")
            f.write(f"Validation MSE Loss: {avg_loss:.4f}\n\n")
        # Example: print first 5 predictions vs targets
        # for i in range(5):
        #     print(f"Pred: {preds[i].item():.3f}, Target: {targets[i].item():.3f}")