import json
import numpy as np
import math

from position import encode_move, encode_board_with_history


with open("move_pairs.json") as f:
    move_pairs = json.load(f)

VALUE_SCALE = 400

boards = []
from_labels = []
to_labels = []
promo_labels = []
value_labels = []

for pair in move_pairs:
    board_tensor = encode_board_with_history(pair["fen"], pair["history"], pair["my_color"])
    move_labels = encode_move(pair["move"], pair["my_color"])

    boards.append(board_tensor)
    from_labels.append(move_labels["from_square"])
    to_labels.append(move_labels["to_square"])
    promo_labels.append(move_labels["promotion"])
    
    normalized_value = math.tanh(pair["centipawn_score"]/VALUE_SCALE)
    value_labels.append(normalized_value)

boards = np.array(boards, dtype=np.float32)
from_labels = np.array(from_labels, dtype=np.int64)
to_labels = np.array(to_labels, dtype=np.int64)
promo_labels = np.array(promo_labels, dtype=np.int64)
value_labels = np.array(value_labels, dtype=np.float32)

np.savez("dataset.npz", boards=boards, from_labels=from_labels, to_labels=to_labels, promo_labels=promo_labels, value_labels=value_labels)
print(f"Saved dataset.npz with {len(boards)} samples")
print(f"boards shape: {boards.shape}, from_labels shape: {from_labels.shape}, to_labels shape: {to_labels.shape}, promo_labels shape: {promo_labels.shape}")
print(f"promo_labels: {np.bincount(promo_labels)}")
print(f"value_labels range: [{value_labels.min():.3f}, {value_labels.max():.3f}], mean: {value_labels.mean():.3f}")