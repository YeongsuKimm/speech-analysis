import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from StreamerDatasetReg import StreamerDataset, MultiModalMLP

def evaluate_model(model_path, dataset, batch_size=16, device="cuda"):
    """
    Evaluates a trained model on a dataset and prints performance metrics.
    
    Args:
        model_path (str): Path to the trained model file.
        dataset (Dataset): The dataset to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to run the evaluation on ("cuda" or "cpu").
    
    Returns:
        dict: Dictionary containing MSE, MAE, and R² score.
    """
    # Define model parameters
    text_dim = 768  
    audio_dim = 768  
    hidden_dim = 128
    output_dim = 1  

    # Load the model
    model = MultiModalMLP(text_dim, audio_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create DataLoader for evaluation
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for text_features, audio_features, labels in dataloader:
            text_features, audio_features, labels = (
                text_features.to(device),
                audio_features.to(device),
                labels.to(device),
            )

            outputs = model(text_features, audio_features).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute evaluation metrics
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"Evaluation Results for {model_path}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")

    return {"MSE": mse, "MAE": mae, "R2": r2}


test_dataset = StreamerDataset(".test/")  # Replace with actual test dataset
model_path = f"models/streamer_reg_model_fold1.pth"
results = evaluate_model(model_path, test_dataset)
