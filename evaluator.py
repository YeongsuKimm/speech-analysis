import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Evaluation function with additional diagnostics
def evaluate_model(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for text_features, audio_features, metadata_features in dataloader:
            text_features, audio_features, metadata_features = (
                text_features.to(device), 
                audio_features.to(device), 
                metadata_features.to(device)
            )
            
            # Get predictions
            outputs = model(text_features, audio_features)  # Shape: [batch_size, 10] (log-scale prediction)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert to class labels
            
            # Store values
            all_predictions.extend(predictions)
            all_targets.extend(metadata_features.cpu().numpy())  # True log-scaled followers

    # Convert lists to NumPy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # Compute metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print(f"\nEvaluation Results:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - R² Score: {r2:.4f}\n")

    # Confusion Matrix
    unique_classes = np.unique(all_targets)  # Extract class labels
    plot_confusion_matrix(all_targets, all_predictions, classes=unique_classes)

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_targets, all_predictions, digits=4))

    return mae, r2


# Load test dataset
test_dataset = StreamerDataset("data/")  # Replace with actual test folder
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load trained model (if needed)
model.load_state_dict(torch.load("multimodal_mlp.pth"))  # Replace with your model path

# Run evaluation
evaluate_model(model, test_dataloader)
