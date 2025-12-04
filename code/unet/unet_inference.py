import torch, cv2, numpy as np
from pathlib import Path
from typing import Literal
from unet import Unet

def infer_all(model_pth: str, images_root: str, save_root: str,
              device: Literal['cpu','cuda'] = 'cuda',
              channels=(32,64,128,256), thresh: float = 0.5):
    model = Unet(1, 1, list(channels)).to(device).eval()
    state = torch.load(model_pth, map_location=device)
    state = state.get("model_state", state.get("state_dict", state))
    state = {k[7:] if k.startswith("module.") else k: v for k,v in state.items()}
    model.load_state_dict(state, strict=True)

    for tif in Path(images_root).rglob("*.tif"):
        img = cv2.imread(str(tif), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        x = torch.from_numpy(img)[None,None].to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
        print(tif.name, float(prob.min()), float(prob.mean()), float(prob.max()))
        mask = (prob > thresh).astype(np.uint8)*255
        outp = Path(save_root) / tif.relative_to(images_root)
        outp = outp.with_suffix(".png")
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), mask)