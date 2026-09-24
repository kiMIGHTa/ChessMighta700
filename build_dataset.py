import json
import numpy as np

from position import encode_move, encode_board_with_history


with open("move_pairs.json") as f:
    move_pairs = json.load(f)

boards = []
from_labels = []
to_labels = []
promo_labels = []

for pair in move_pairs:
    board_tensor = encode_board_with_history(pair["fen"], pair["history"], pair["my_color"])
    move_labels = encode_move(pair["move"], pair["my_color"])

    boards.append(board_tensor)
    from_labels.append(move_labels["from_square"])
    to_labels.append(move_labels["to_square"])
    promo_labels.append(move_labels["promotion"])

boards = np.array(boards, dtype=np.float32)
from_labels = np.array(from_labels, dtype=np.int64)
to_labels = np.array(to_labels, dtype=np.int64)
promo_labels = np.array(promo_labels, dtype=np.int64)

np.savez("dataset.npz", boards=boards, from_labels=from_labels, to_labels=to_labels, promo_labels=promo_labels)
print(f"Saved dataset.npz with {len(boards)} samples")
print(f"boards shape: {boards.shape}, from_labels shape: {from_labels.shape}, to_labels shape: {to_labels.shape}, promo_labels shape: {promo_labels.shape}")
print(f"promo_labels: {np.bincount(promo_labels)}")