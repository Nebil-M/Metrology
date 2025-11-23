import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)


class ImageMaskRNNClassifier(nn.Module):
    """
    RNN-based classifier over (image, mask) pair.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels,
                                  base_channels=base_channels)

        self.rnn = nn.GRU(
            input_size=self.encoder.out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f_img = self.encoder(img)    # (B,F)
        f_mask = self.encoder(mask)  # (B,F)
        seq = torch.stack([f_img, f_mask], dim=1)  # (B,2,F)

        _, h_n = self.rnn(seq)
        h_last = h_n[-1]            # (B,H)
        logits = self.fc(h_last)    # (B,1)
        return logits.squeeze(1)    # (B,)
