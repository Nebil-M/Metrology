import torch, torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
        targets = targets.to(logits.dtype)
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        dice = 1 - (2*(p*targets).sum((2,3)) / ((p+targets).sum((2,3)) + eps)).mean()
        return bce + dice
