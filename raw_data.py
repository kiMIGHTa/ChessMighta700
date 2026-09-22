import json
import requests

USERNAME = "kimighta"
headers = {"User-Agent": "MyChessBot (dkimaita22@gmail.com)"}

archives = requests.get(
    "https://api.chess.com/pub/player/kimighta/games/archives", headers=headers).json()["archives"]


# keep 2025-2026
archives = [url for url in archives if "2025" in url or "2026" in url]

games_data = []

for url in archives:
    monthly_games = requests.get(url, headers=headers).json()["games"]
    for g in monthly_games:
        if g.get("rules") != "chess" or g.get("rated")!= True:
            continue
        if g.get("time_class") != "rapid":
            continue

        white_player = g["white"]["username"].lower()
        black_player = g["black"]["username"].lower()

        if white_player == USERNAME:
            my_color, my_rating, my_result =  "white", g["white"]["rating"], g["white"]["result"] 
        elif black_player == USERNAME:
            my_color, my_rating, my_result = "black", g["black"]["rating"], g["black"]["result"]
        else:
            continue

        games_data.append({
                "pgn": g["pgn"],
                "my_color": my_color,
                "my_rating": my_rating,
                "my_result": my_result,
                "end_time": g["end_time"],
            })
        

print(f"Collected {len(games_data)} rated rapid games (2025-2026)")

with open("raw_games.json", "w") as f:
    json.dump(games_data, f, indent =4)

# if response.status_code == 200:
#     data = response.json()
#     print(data)
# else:
#     print(f"Failed to fetch data. Status code: {response.status_code}")
