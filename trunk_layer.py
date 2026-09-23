import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.3)  # dropout layer with a probability of 0.5 to prevent overfitting — randomly sets 50% of the input units to 0 during training, which helps the model generalize better to unseen data

        # output heads
        self.from_head = nn.Linear(128 * 8 * 8, 64)
        self.to_head = nn.Linear(128 * 8 * 8, 64)
        self.promo_head = nn.Linear(128 * 8 * 8, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x) # inserts non-linearity after the first convolution otherwise conv1 output would be a linear function of the input, which is not very useful for learning complex patterns
        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # applies dropout during training

        from_logits = self.from_head(x)
        to_logits = self.to_head(x)
        promo_logits = self.promo_head(x)

        return (from_logits, to_logits, promo_logits)
