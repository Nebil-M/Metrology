import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ================= CONFIGURATION =================
# Path to the balanced dataset created by the previous script
DATASET_DIR = "Evaluator_Dataset_balanced"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 40 #Best 30
# =================================================

class WaferEvaluatorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # List of (img_path, mask_path, label)

        # 1. Load Good Samples (Label = 1)
        good_img_dir = os.path.join(root_dir, "1_Good", "images")
        good_msk_dir = os.path.join(root_dir, "1_Good", "masks")
        self._load_class_samples(good_img_dir, good_msk_dir, label=1)

        # 2. Load Bad Samples (Label = 0)
        bad_img_dir = os.path.join(root_dir, "0_Bad", "images")
        bad_msk_dir = os.path.join(root_dir, "0_Bad", "masks")
        self._load_class_samples(bad_img_dir, bad_msk_dir, label=0)

        print(f"-> Found {len(self.samples)} pairs in total.")

    def _load_class_samples(self, img_dir, msk_dir, label):
        if not os.path.exists(img_dir): return
        
        # Get all images
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]

        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            
            # Try to find corresponding mask (name match)
            stem = os.path.splitext(img_name)[0]
            
            # Heuristic: Find mask file with same stem
            mask_name = None
            for f in os.listdir(msk_dir):
                if f.startswith(stem) and f.lower().endswith(valid_exts):
                    mask_name = f
                    break
            
            if mask_name:
                mask_path = os.path.join(msk_dir, mask_name)
                self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        # Open Image (Convert to RGB to standardize inputs)
        image = Image.open(img_path).convert("RGB")
        
        # Open Mask (Convert to Grayscale/L)
        mask = Image.open(mask_path).convert("L")

        # Resize to standard size (e.g., 224x224 for ResNet)
        # We must apply the same resize to both to keep them aligned
        resize = transforms.Resize((224, 224))
        image = resize(image)
        mask = resize(mask)

        # Convert to Tensor
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image)   # Shape: (3, 224, 224)
        msk_tensor = to_tensor(mask)    # Shape: (1, 224, 224)

        # Concatenate along channel dimension -> (4, 224, 224)
        # Input is now: R, G, B, Mask
        input_tensor = torch.cat([img_tensor, msk_tensor], dim=0)
        
        # Normalize (Optional but recommended - usually done on 3 channels, 
        # here we skip specific norm for simplicity or can add custom 4-channel norm)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return input_tensor, label_tensor

def get_evaluator_model():
    """
    Loads ResNet18 and modifies the first layer to accept 4 channels 
    (Image RGB + Mask) instead of 3.
    """
    # Load pre-trained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # --- Modify First Conv Layer ---
    # Original: Conv2d(3, 64, kernel_size=7, ...)
    # New:      Conv2d(4, 64, kernel_size=7, ...)
    original_conv = model.conv1
    
    # Create new layer with 4 input channels
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Initialize weights:
    # Copy original RGB weights to first 3 channels
    with torch.no_grad():
        new_conv.weight[:, :3] = original_conv.weight
        # For the 4th channel (mask), we can initialize it with the mean of the RGB weights
        # This gives it a "reasonable" starting point
        new_conv.weight[:, 3] = torch.mean(original_conv.weight, dim=1)

    model.conv1 = new_conv

    # --- Modify Final Fully Connected Layer ---
    # ResNet output is 1000 classes (ImageNet). We need 1 output (Binary: 0 or 1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model

def train_model():
    print(f"Loading dataset from: {DATASET_DIR}")
    
    dataset = WaferEvaluatorDataset(DATASET_DIR)
    
    if len(dataset) == 0:
        print("Error: No valid image/mask pairs found. Check your paths.")
        return

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[Quality] Training Samples: {len(train_dataset)}")
    print(f"[Quality] Val Samples:      {len(val_dataset)}")

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Quality] Training on {device}")

    # Initialize Model
    model = get_evaluator_model().to(device)

    # --- DYNAMIC CLASS WEIGHTING ---
    # Calculate ratio from the dataset itself so we don't need to hardcode it
    all_labels = [s[2] for s in dataset.samples]
    num_good = sum(all_labels)
    num_bad = len(all_labels) - num_good

    if num_good > 0:
        pos_weight_val = num_bad / num_good
    else:
        pos_weight_val = 1.0 # Fallback if no good samples
    
    print(f"[Balance] Good: {int(num_good)} | Bad: {int(num_bad)} | Calculated pos_weight: {pos_weight_val:.2f}")

    # Apply the calculated weight to the loss function
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs).squeeze() # Output shape (Batch_Size)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        print(f"   -> Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_evaluator_model_1.pth")
            print("   -> Model Saved!")

    print("\nTraining Complete. Best Model saved as 'best_evaluator_unbalanced.pth'")

if __name__ == "__main__":
    train_model()