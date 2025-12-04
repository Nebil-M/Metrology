import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# ================= CONFIGURATION =================
TEST_DATASET_DIR = r"C:\Repo\Metrology\Evaluator_Dataset_Test copy"
MODEL_PATH = "Balanced_evalutor.pth"
BATCH_SIZE = 16
# =================================================

class WaferEvaluatorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = [] 

        # 1. Load Good Samples (Label = 1)
        good_img_dir = os.path.join(root_dir, "1_Good", "images")
        good_msk_dir = os.path.join(root_dir, "1_Good", "masks")
        self._load_class_samples(good_img_dir, good_msk_dir, label=1)

        # 2. Load Bad Samples (Label = 0)
        bad_img_dir = os.path.join(root_dir, "0_Bad", "images")
        bad_msk_dir = os.path.join(root_dir, "0_Bad", "masks")
        self._load_class_samples(bad_img_dir, bad_msk_dir, label=0)

    def _load_class_samples(self, img_dir, msk_dir, label):
        if not os.path.exists(img_dir): return
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        try:
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        except OSError: return

        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            stem = os.path.splitext(img_name)[0]
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

        # Open Image (RGB) & Mask (L)
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data to prevent crash
            return torch.zeros(4, 224, 224), torch.tensor(label, dtype=torch.float32), idx

        resize = transforms.Resize((224, 224))
        image = resize(image)
        mask = resize(mask)

        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image)
        msk_tensor = to_tensor(mask)

        input_tensor = torch.cat([img_tensor, msk_tensor], dim=0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return input_tensor, label_tensor, idx

def get_evaluator_model():
    model = models.resnet18(weights=None)
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = new_conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

class EvaluatorTester:
    def __init__(self, dataset_dir, model_path, batch_size=16):
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing Tester on {self.device}...")
        self.dataset = WaferEvaluatorDataset(self.dataset_dir)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model = self._load_model()
        
        # Storage for inference results
        self.all_labels = []
        self.all_probs = []
        self.all_losses = []
        self.sample_indices = [] 

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        model = get_evaluator_model().to(self.device)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
        model.eval()
        return model

    def run_inference(self):
        """Runs the model on the dataset and caches probabilities and losses."""
        print(f"--- Running Inference on {len(self.dataset)} samples ---")
        criterion = nn.BCEWithLogitsLoss(reduction='none') 

        with torch.no_grad():
            for inputs, labels, indices in self.loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs).squeeze()
                
                # Per-sample loss
                batch_loss = criterion(outputs, labels)
                
                # Probabilities (0.0 to 1.0)
                probs = torch.sigmoid(outputs)
                
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
                self.all_losses.extend(batch_loss.cpu().numpy())
                self.sample_indices.extend(indices.numpy())
        print("Inference Complete.\n")

    def generate_report(self, thresholds):
        """Generates detailed reports for specific thresholds."""
        if not self.all_probs:
            print("Please run_inference() first.")
            return

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        losses = np.array(self.all_losses)

        # 1. Build Structure Map
        structure_map = [] 
        for idx in self.sample_indices:
            path = self.dataset.samples[idx][0]
            filename = os.path.basename(path)
            # Assumes "Structure_Filename.tif" or just takes parent folder name logic if needed
            # Here we split by "_" as per previous conventions
            if "_" in filename:
                struct_name = filename.split("_")[0]
            else:
                struct_name = "Unknown"
            structure_map.append(struct_name)
        
        structure_map = np.array(structure_map)
        unique_structures = np.unique(structure_map)

        # 2. Global Loss Report
        print("="*120)
        print(f"  GLOBAL PERFORMANCE REPORT")
        print("="*120)
        print(f"Total Samples: {len(labels)}")
        print(f"Average Binary Cross Entropy Loss: {np.mean(losses):.4f}")
        print("-" * 120)

        # 3. Threshold Loop
        for thresh in thresholds:
            preds = (probs > thresh).astype(float)
            
            print(f"\n>>> ANALYSIS AT CONFIDENCE THRESHOLD: {thresh}")
            print("-" * 80)
            
            # --- Global Metrics ---
            self._print_metrics(labels, preds, "Global")

            # --- Per-Structure Breakdown ---
            print("\n  --- Per-Structure Breakdown ---")
            # Headers
            # TP: True Positive (Correctly identified Good)
            # TN: True Negative (Correctly rejected Bad)
            # FP: False Positive (Bad labeled Good - Poison)
            # FN: False Negative (Good labeled Bad - Waste)
            print(f"  {'Structure':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'True Pos':<18} {'True Neg':<18} {'False Pos':<20} {'False Neg':<20}")
            print("  " + "-"*130)

            for struct in unique_structures:
                mask = (structure_map == struct)
                if np.sum(mask) == 0: continue
                
                s_labels = labels[mask]
                s_preds = preds[mask]
                total_struct = len(s_labels)
                
                # Standard Metrics
                acc = accuracy_score(s_labels, s_preds)
                prec = precision_score(s_labels, s_preds, zero_division=0)
                rec = recall_score(s_labels, s_preds, zero_division=0)
                
                # Confusion Matrix for Counts
                # force labels=[0,1] so we always get a 2x2 matrix even if a class is missing
                tn, fp, fn, tp = confusion_matrix(s_labels, s_preds, labels=[0,1]).ravel()
                
                # Percentages relative to total structure count
                tp_pct = (tp / total_struct) * 100 if total_struct > 0 else 0
                tn_pct = (tn / total_struct) * 100 if total_struct > 0 else 0
                fp_pct = (fp / total_struct) * 100 if total_struct > 0 else 0
                fn_pct = (fn / total_struct) * 100 if total_struct > 0 else 0
                
                # Format: "Count/Total (Pct%)"
                tp_str = f"{tp}/{total_struct} ({tp_pct:.0f}%)"
                tn_str = f"{tn}/{total_struct} ({tn_pct:.0f}%)"
                fp_str = f"{fp}/{total_struct} ({fp_pct:.0f}%)"
                fn_str = f"{fn}/{total_struct} ({fn_pct:.0f}%)"

                print(f"  {struct:<12} {acc*100:.1f}%      {prec:.4f}     {rec:.4f}     {tp_str:<18} {tn_str:<18} {fp_str:<20} {fn_str:<20}")
            print("-" * 130)

    def _print_metrics(self, y_true, y_pred, title):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        
        total = len(y_true)
        total_bad = tn + fp # Actual Negatives
        total_good = tp + fn # Actual Positives
        
        # Rates
        fpr = fp / total_bad if total_bad > 0 else 0.0
        fnr = fn / total_good if total_good > 0 else 0.0

        print(f"  [{title}]")
        print(f"  Accuracy:           {acc*100:.2f}%")
        print(f"  Precision:          {prec:.4f}  (Trustworthiness)")
        print(f"  Recall:             {rec:.4f}   (Yield)")
        print(f"  F1 Score:           {f1:.4f}")
        print(f"  Confusion Matrix:   TP={tp} | TN={tn} | FP={fp} | FN={fn}")
        print(f"  False Positive Rate: {fpr*100:.2f}% ({fp}/{total_bad} Bad masks incorrectly labeled Good)")
        print(f"  False Negative Rate: {fnr*100:.2f}% ({fn}/{total_good} Good masks incorrectly rejected)")


if __name__ == "__main__":
    # Initialize the Tester
    tester = EvaluatorTester(
        dataset_dir=TEST_DATASET_DIR, 
        model_path=MODEL_PATH, 
        batch_size=16
    )
    
    # Run Inference once (this takes the most time)
    tester.run_inference()

    # --- Scenario 1: Standard Analysis ---
    # Look at a few spread out thresholds to see how the model behaves
    print("\n" + "="*20 + " SCENARIO 1: Standard Sweep " + "="*20)
    tester.generate_report([0.5, 0.7, 0.9])

    # --- Scenario 2: High Precision Curation ---
    # This is what you likely want for your Auto-Curator.
    # We look for the threshold where False Positives are near zero.
    print("\n" + "="*20 + " SCENARIO 2: Strict Curation " + "="*20)
    tester.generate_report([0.95, 0.98, 1.00])