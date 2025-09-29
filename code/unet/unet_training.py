import torch
from torch.utils.data import DataLoader, random_split
from typing import Literal
from dataset import SEMDataset
from unet import Unet
from losses import BCEDiceLoss
from datetime import datetime

def train_stage(data_root: str, model_output: str, epochs: int = 50, lr: float = 1e-4, bs: int = 8,
                device: Literal['cpu','cuda'] = 'cpu', channels=(32,64,128,256)):
    
    print(f"Training on {device} with {len(channels)} channels: {channels}")
    
    imgs, lbls = f"{data_root}/image", f"{data_root}/label"
    ds = SEMDataset(imgs, lbls)
    n_val = max(1, int(0.2*len(ds)))
    
    g = torch.Generator().manual_seed(0)
    tr_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=g)

    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=bs, pin_memory=True, num_workers=4)

    

    model = Unet(1, 1, list(channels)).to(device)
    crit  = BCEDiceLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best = float('inf')
    for ep in range(epochs):
        model.train()
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for x,y in val_ld:
                x,y = x.to(device), y.to(device)
                tot += crit(model(x), y).item()
        avg = tot/len(val_ld)


        ############## Logging
        msg = f"[{datetime.now().isoformat(timespec='seconds')}] Epoch {ep+1}/{epochs} val_loss={avg:.4f}"
        print(msg, flush=True)
        with open(r"C:/Repo/Metrology/models/log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        ##############


        print(f"[Epoch {ep+1}/{epochs}] val_loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), model_output)
            print(f"→ Saved {model_output}")
