import torch
import chess
from trunk_layer import ChessNet
from position import PROMO_TO_LABEL, encode_board

model = ChessNet()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

def predict_move(fen, my_color):
    board = chess.Board(fen)
    board_tensor = encode_board(fen, my_color)

    # model expects a batch dimension — (batch, 12, 8, 8), not (12, 8, 8)
    # convert board_tensor (currently a numpy array) to a torch tensor and add a batch dim
    input_tensor = torch.from_numpy(board_tensor).unsqueeze(0)

    with torch.no_grad():
        from_logits, to_logits, promo_logits = model(input_tensor)

    # get the predicted square/class from each head
    # from_square = from_logits.argmax(dim=1).item()
    # to_square = to_logits.argmax(dim=1).item()
    # promo_class = promo_logits.argmax(dim=1).item()

    # squeeze out the batch dim so we have plain (64,) tensors to index into
    from_scores = from_logits.squeeze(0)
    to_scores = to_logits.squeeze(0)
    promo_scores = promo_logits.squeeze(0)

    best_move = None
    best_score = float("-inf")

    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        promo = move.promotion

    # if my_color == "black", mirror the squares back to the original orientation
        if my_color == "black":
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        promo_label = PROMO_TO_LABEL[promo]

        
        move_score = from_scores[from_sq] + to_scores[to_sq] + promo_scores[promo_label]

        if move_score > best_score:
            best_score = move_score
            best_move = move

    print(f"Predicted best move: {best_move} with score {best_score.item():.4f}")

predict_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "black")