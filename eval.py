import torch
import numpy as np
from trunk_layer import ChessNet
from training_model import val_dataset  # reuse the same fixed split

model = ChessNet()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

correct_from = 0
correct_to = 0
correct_both = 0
total = 0

with torch.no_grad():
    for board, from_label, to_label, promo_label in val_dataset:
        input_tensor = board.unsqueeze(0)
        from_logits, to_logits, promo_logits = model(input_tensor)

        pred_from = np.argmax(from_logits).item()
        pred_to = np.argmax(to_logits).item()

        if pred_from == from_label.item():
            correct_from += 1
        if pred_to == to_label.item():
            correct_to += 1
        if pred_from == from_label.item() and pred_to == to_label.item():
            correct_both += 1
        total += 1

print(f"From-square accuracy: {correct_from/total:.2%}")
print(f"To-square accuracy: {correct_to/total:.2%}")
print(f"Full-move accuracy: {correct_both/total:.2%}")