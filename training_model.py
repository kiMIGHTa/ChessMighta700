import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ChessDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.boards = data["boards"]
        self.from_labels = data["from_labels"]
        self.to_labels = data["to_labels"]
        self.promo_labels = data["promo_labels"]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board_tensor = torch.from_numpy(self.boards[idx])
        from_label = torch.tensor(self.from_labels[idx])
        to_label = torch.tensor(self.to_labels[idx])
        promo_label = torch.tensor(self.promo_labels[idx])

        return board_tensor, from_label, to_label, promo_label


dataset = ChessDataset("dataset.npz")


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# wrap the datasets in DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")