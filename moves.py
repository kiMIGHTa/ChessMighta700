import json
import chess
import chess.pgn
import io

with open("raw_games.json") as f:
    games_data = json.load(f)

training_pairs = []  # list of dicts: {"fen": ..., "move": ..., "my_color": ...}
HISTORY_LENGTH = 2  # how many past positions to keep per sample


for game_record in games_data:
    pgn_io = io.StringIO(game_record["pgn"])
    game = chess.pgn.read_game(pgn_io)
    board = game.board()

    my_color = game_record["my_color"]  # "white" or "black"

    fen_history = []  


    for move in game.mainline_moves():
        # Check if it's my player's turn to move
        if (board.turn and my_color == "white") or (not board.turn and my_color == "black"):
           
            history = fen_history[-HISTORY_LENGTH:]

            training_pairs.append({
                "fen": board.fen(),
                "history": history,
                "move": move.uci(),
                "my_color": my_color
            })

        fen_history.append(board.fen())


        board.push(move)  # applies the move, advancing the board

print(f"Extracted {len(training_pairs)} of your own moves")

with open("move_pairs.json", "w") as f:
    json.dump(training_pairs, f, indent=4)