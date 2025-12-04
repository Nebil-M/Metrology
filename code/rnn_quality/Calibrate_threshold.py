import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from PIL import Image

# Import model definition from your trainer
from Refined_Trainer import get_evaluator_model, WaferEvaluatorDataset

# ================= CONFIGURATION =================
# Point this to the VALIDATION folder
VAL_DIR = r"C:\Repo\Metrology\Evaluator_Dataset_Validation_balanced"
MODEL_PATH = "Balanced_evalutor.pth"
BATCH_SIZE = 16
TARGET_PRECISION = 0.98  # Safety requirement (98% precision)
# =================================================

def analyze_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Threshold Analysis on {device} ---")

    # 1. Load Data
    val_dataset = WaferEvaluatorDataset(VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    model = get_evaluator_model().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    model.eval()

    # 3. Get Predictions
    y_true = []
    y_scores = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # =================================================
    # PART A: ROC Curve
    # =================================================
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Setup Figure with 3 Subplots
    plt.figure(figsize=(18, 6))
    
    # Plot 1: ROC
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess (Baseline)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Poison)')
    plt.ylabel('True Positive Rate (Yield)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # =================================================
    # PART B: Precision-Recall Curve
    # =================================================
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Plot 2: PR
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
    plt.xlabel('Recall (Yield)')
    plt.ylabel('Precision (Purity)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    # =================================================
    # PART C: Metrics vs Threshold (The requested graph)
    # =================================================
    # Note: precision and recall arrays have 1 extra element compared to thresholds
    # We slice [:-1] to align them for plotting against thresholds
    
    plt.subplot(1, 3, 3)
    plt.title('Metrics vs. Decision Threshold')
    plt.plot(pr_thresholds, precision[:-1], 'b--', label='Precision (Purity)', linewidth=2)
    plt.plot(pr_thresholds, recall[:-1], 'g-', label='Recall (Yield)', linewidth=2)
    plt.xlabel('Threshold Score (0.0 to 1.0)')
    plt.ylabel('Metric Value')
    plt.legend(loc='center left')
    plt.grid(alpha=0.3, which='both')
    
    # Add vertical line for 0.5 default
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    # =================================================
    # PART D: Find Optimal Threshold
    # =================================================
    valid_indices = np.where(precision[:-1] >= TARGET_PRECISION)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[0]
        best_thresh = pr_thresholds[best_idx]
        best_recall = recall[best_idx]
        actual_prec = precision[best_idx]
        
        # Mark on graphs
        # On PR Curve
        plt.subplot(1, 3, 2)
        plt.plot(best_recall, actual_prec, 'ro', label='Selected Point')
        plt.legend()
        
        # On Threshold Curve
        plt.subplot(1, 3, 3)
        plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Selected T={best_thresh:.2f}')
        plt.legend()
        
        print("\n" + "="*40)
        print(f"THRESHOLD ANALYSIS REPORT")
        print("="*40)
        print(f"Goal:           Min {TARGET_PRECISION*100}% Precision")
        print(f"Recommended T:  {best_thresh:.4f}")
        print("-" * 40)
        print(f"At threshold {best_thresh:.4f}:")
        print(f"  -> Precision: {actual_prec*100:.2f}% (Purity of kept data)")
        print(f"  -> Recall:    {best_recall*100:.2f}% (Percent of good data recovered)")
        print("="*40)
    else:
        print(f"\n[WARN] Model never reached {TARGET_PRECISION*100}% precision.")

    plt.tight_layout()
    plt.savefig("model_performance_curves_detailed.png")
    print("\nGraphs saved to 'model_performance_curves_detailed.png'")
    plt.show()

if __name__ == "__main__":
    analyze_performance()