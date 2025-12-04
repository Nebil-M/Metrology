import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ================= CONFIGURATION =================
# Separate Training and Validation Directories
TRAIN_DIR = r"C:\Repo\Metrology\Evaluator_Dataset" 
VAL_DIR   = r"C:\Repo\Metrology\Evaluator_Dataset_Validation"
MODEL_NAME = r'Un_Balanced_evalutor.pth'

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 40 
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

        print(f"-> Loaded {len(self.samples)} samples from {os.path.basename(root_dir)}")

    def _load_class_samples(self, img_dir, msk_dir, label):
        if not os.path.exists(img_dir): return
        
        # Get all images
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        try:
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        except OSError: return

        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            
            # Try to find corresponding mask (name match)
            stem = os.path.splitext(img_name)[0]
            
            # Heuristic: Find mask file with same stem
            mask_name = None
            if os.path.exists(msk_dir):
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
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy to prevent crash
            return torch.zeros(4, 224, 224), torch.tensor(label, dtype=torch.float32)

        # Resize to standard size (224x224 for ResNet)
        resize = transforms.Resize((224, 224))
        image = resize(image)
        mask = resize(mask)

        # Convert to Tensor
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image)   # Shape: (3, 224, 224)
        msk_tensor = to_tensor(mask)    # Shape: (1, 224, 224)

        # Concatenate along channel dimension -> (4, 224, 224)
        input_tensor = torch.cat([img_tensor, msk_tensor], dim=0)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return input_tensor, label_tensor

def get_evaluator_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    original_conv = model.conv1
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        new_conv.weight[:, :3] = original_conv.weight
        new_conv.weight[:, 3] = torch.mean(original_conv.weight, dim=1)

    model.conv1 = new_conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def train_model():
    print(f"--- Setting up Datasets ---")
    
    # 1. Initialize Separate Datasets
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("Error: Train or Val directory not found. Please run the splitting script first.")
        return

    train_dataset = WaferEvaluatorDataset(TRAIN_DIR)
    val_dataset   = WaferEvaluatorDataset(VAL_DIR)
    
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty.")
        return

    # 2. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Initialize Model
    model = get_evaluator_model().to(device)

    # 3. Calculate Class Weighting (Based on TRAINING set only)
    all_train_labels = [s[2] for s in train_dataset.samples]
    num_good = sum(all_train_labels)
    num_bad = len(all_train_labels) - num_good

    # Avoid division by zero
    if num_good > 0:
        pos_weight_val = num_bad / num_good
    else:
        pos_weight_val = 1.0 
    
    print(f"[Class Balance] Good: {int(num_good)} | Bad: {int(num_bad)}")
    print(f"[Class Balance] Calculated pos_weight: {pos_weight_val:.4f}")

    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_acc = 0.0
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation Phase
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
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_NAME)
            print(f"   >>> Best Model Saved (Val Acc: {val_acc:.2f}%)")

    print("\nTraining Complete.")

if __name__ == "__main__":
    train_model()