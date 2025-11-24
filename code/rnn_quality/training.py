import torch
from torch.utils.data import DataLoader, random_split
from typing import Literal
from datetime import datetime

# Make sure this imports the NEW dataset class we just defined
from dataset import QualityDataset
from model import ImageMaskRNNClassifier

def train_quality_stage(
    data_root: str,
    model_output: str,
    epochs: int = 50,
    lr: float = 1e-4,
    bs: int = 8,
    device: Literal["cpu", "cuda"] = "cuda",
    base_channels: int = 16,
    rnn_hidden: int = 64,
    bidirectional: bool = False,
):
    print(f"[Quality] Initializing dataset from {data_root}...")

    # Instantiate the new folder-based dataset
    ds = QualityDataset(data_root)
    
    n_val = max(1, int(0.2 * len(ds)))
    g = torch.Generator().manual_seed(0)
    tr_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)

    print(f"[Quality] Training Samples: {len(tr_ds)}")
    print(f"[Quality] Val Samples:      {len(val_ds)}")

    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                       pin_memory=False, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=bs,
                        pin_memory=False, num_workers=4)

    device_t = torch.device(device if (device == "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[Quality] Training on {device_t}")

    model = ImageMaskRNNClassifier(
        in_channels=1,
        base_channels=base_channels,
        rnn_hidden=rnn_hidden,
        bidirectional=bidirectional,
    ).to(device_t)

    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for ep in range(epochs):
        # ---- train ----
        model.train()
        running = 0.0
        for img, mask, y in tr_ld:
            img, mask, y = img.to(device_t), mask.to(device_t), y.to(device_t)
            opt.zero_grad()
            logits = model(img, mask)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * img.size(0)
        tr_loss = running / len(tr_ds)

        # ---- val ----
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for img, mask, y in val_ld:
                img, mask, y = img.to(device_t), mask.to(device_t), y.to(device_t)
                logits = model(img, mask)
                loss = crit(logits, y)
                val_running += loss.item() * img.size(0)
        val_loss = val_running / len(val_ds)

        msg = f"[{datetime.now().isoformat(timespec='seconds')}] " \
              f"Epoch {ep+1}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}"
        print(msg, flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, model_output)
            print(f"  -> Saved best model to {model_output}")