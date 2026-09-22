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
    tensor = np.zeros((17, 8, 8), dtype=np.float32) #17 planes: 12 for pieces, 4 for castling rights, 1 for en passant square
    my_chess_color = chess.WHITE if my_color == "white" else chess.BLACK
    opponent_color = not my_chess_color


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


    tensor[12, :, :] = 1 if board.has_kingside_castling_rights(my_chess_color) else 0

    tensor[13, :, :] = 1 if board.has_queenside_castling_rights(my_chess_color) else 0

    tensor[14, :, :] = 1 if board.has_kingside_castling_rights(opponent_color) else 0

    tensor[15, :, :] = 1 if board.has_queenside_castling_rights(opponent_color) else 0

    if board.ep_square is not None:
        ep_square = chess.square_mirror(board.ep_square) if my_color == "black" else board.ep_square
        rank = chess.square_rank(ep_square)
        file = chess.square_file(ep_square)
        tensor[16][rank][file] = 1

    return tensor



def encode_move(move_uci, my_color):
    move = chess.Move.from_uci(move_uci)

    from_square = move.from_square
    to_square = move.to_square

    # flip the squares if my_color == "black" (mirror them)
    if my_color == "black":
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    

    promotion_label = PROMO_TO_LABEL.get(move.promotion, 0)

    return {
        "from_square": from_square,
        "to_square": to_square,
        "promotion": promotion_label
    }