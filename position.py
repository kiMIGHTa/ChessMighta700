import chess
import numpy as np

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PROMO_TO_LABEL = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

def encode_board(fen, my_color):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    my_chess_color = chess.WHITE if my_color == "white" else chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # base plane (0-5) for this piece's TYPE using PIECE_TO_PLANE
        plane_index = PIECE_TO_PLANE.get(piece.piece_type)

        # plane offset — 0 if piece.color == my_chess_color (mine),
        # otherwise 6 (opponent's). Add this offset to the base plane.
        plane_offset = 0 if piece.color == my_chess_color else 6
        plane_index += plane_offset

        #  get the square to use for indexing —
        #  if my_color == "black", use chess.square_mirror(square), otherwise use square as-is
        indexed_square = chess.square_mirror(square) if my_color == "black" else square

        #  get rank and file from that (possibly mirrored) square,
        #  then set tensor[plane_index][rank][file] = 1
        rank = chess.square_rank(indexed_square)
        file = chess.square_file(indexed_square)
        tensor[plane_index][rank][file] = 1

    return tensor



def encode_move(move_uci, my_color):
    move = chess.Move.from_uci(move_uci)

    # TODO 1: get move.from_square and move.to_square
    from_square = move.from_square
    to_square = move.to_square

    # TODO 2: if my_color == "black", mirror both squares with chess.square_mirror()
    # otherwise leave them as-is
    if my_color == "black":
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    

    # TODO 3: look up the promotion label using PROMO_TO_LABEL and move.promotion
    promotion_label = PROMO_TO_LABEL.get(move.promotion, 0)

    # TODO 4: return a dict: {"from_square": ..., "to_square": ..., "promotion": ...}
    return {
        "from_square": from_square,
        "to_square": to_square,
        "promotion": promotion_label
    }