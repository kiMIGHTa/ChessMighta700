import torch
import chess
from trunk_layer import ChessNet
from position import PROMO_TO_LABEL, encode_board_with_history

model = ChessNet()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()


PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

# tactical evaluation function to check if a move leaves a piece hanging (undefended) or allows a bad trade. This is used to filter out moves that are tactically unsound, even if the model predicts them as high-scoring moves.
# def is_hanging(board, my_color):
#     """After a move has been made (board reflects the new position, opponent to move),
#     check if the opponent can capture one of my pieces for a bad trade."""
#     opponent_color = not my_color

#     for move in board.legal_moves:
#         if not board.is_capture(move):
#             continue

#         captured_square = move.to_square
#         captured_piece = board.piece_at(captured_square)
#         if captured_piece is None or captured_piece.color != my_color:
#             continue  # not capturing my piece

#         captured_value = PIECE_VALUES[captured_piece.piece_type]
#         attacker_piece = board.piece_at(move.from_square)
#         attacker_value = PIECE_VALUES[attacker_piece.piece_type]
#         my_defenders = board.attackers(my_color, captured_square)

#         # if there are no defenders of my piece, it's a hanging piece
#         if not my_defenders:
#             # nothing defends this square at all — a free capture
#             return True

#         # if the attacker is of lower value than the captured piece, it's a bad trade
#         if attacker_value < captured_value:
#             return True

#     return False


def get_top_candidates(board, my_color, top_k=3):
    
    board_tensor = encode_board_with_history(board.fen(),[], my_color)

    # model expects a batch dimension — (batch, 17, 8, 8), not (17, 8, 8)
    # convert board_tensor (currently a numpy array) to a torch tensor and add a batch dim
    input_tensor = torch.from_numpy(board_tensor).unsqueeze(0)

    with torch.no_grad():
        from_logits, to_logits, promo_logits, _ = model(input_tensor)

    # get the predicted square/class from each head
    # squeeze out the batch dim so we have plain (64,) tensors to index into
    from_scores = from_logits.squeeze(0)
    to_scores = to_logits.squeeze(0)
    promo_scores = promo_logits.squeeze(0)

    scored_moves = []

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
        scored_moves.append((move_score.item(), move))

    # sort the moves by score in descending order and take the top_k moves
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in scored_moves[:top_k]]



def minimax_search(board, my_color, depth, maximizing):
    if depth == 0 or board.is_game_over():
        board_tensor = encode_board_with_history(board.fen(), [], my_color)
        input_tensor = torch.from_numpy(board_tensor).unsqueeze(0)
        with torch.no_grad():
            _, _, _, value_pred = model(input_tensor)
        return value_pred.item()

    candidates = get_top_candidates(board, my_color, top_k=3)

    if maximizing:
        best_value = float("-inf")
        for move in candidates:
            board_copy = board.copy()
            board_copy.push(move)
            value = minimax_search(board_copy, my_color, depth - 1, False)
            best_value = max(best_value, value)
        return best_value
    else:
        best_value = float("inf")
        for move in candidates:
            board_copy = board.copy()
            board_copy.push(move)
            value = minimax_search(board_copy, my_color, depth - 1, True)
            best_value = min(best_value, value)
        return best_value


def predict_move(fen, my_color, search_depth=2):
    board = chess.Board(fen)
    candidates = get_top_candidates(board, my_color, top_k=3)

    best_move = None
    best_value = float("-inf")

    for move in candidates:
        board_copy = board.copy()
        board_copy.push(move)
        value = minimax_search(board_copy, my_color, search_depth - 1, False)  # opponent's turn next
        if value > best_value:
            best_value = value
            best_move = move

    return best_move

def play_game(model_plays="black"):
    board = chess.Board()

    while not board.is_game_over():
        print(board)
        print()

        if (board.turn == chess.WHITE and model_plays == "white") or \
           (board.turn == chess.BLACK and model_plays == "black"):
            move = predict_move(board.fen(), model_plays)
            print(f"Model plays: {move}")
            board.push(move)
        else:
            user_input = input("Your move (e.g. e2e4): ")
            try:
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Try again.")

    print(board)
    print(f"Game over: {board.result()}")

play_game(model_plays="black")