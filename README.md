# botmighta

A chess engine trained to play like *me* — not to play well. It's built from ~24,000 of my own rated rapid games on chess.com (2025–2026, ~700 Elo), and is deployed as a live [Lichess BOT account](https://lichess.org) anyone can challenge.

The goal isn't strength. It's a snapshot: what did a 700-rated version of me actually do at the board — openings, habits, blunders included? Future milestones (1000, 1500, 2000 Elo) are meant to get their own snapshot bots as more games get played.

## How it works

**Pipeline:** chess.com API → PGN parsing → board/move encoding → CNN training → minimax search → Lichess deployment.

1. **`raw_data.py`** — pulls all rated rapid games (2025–2026) via the chess.com public API.
2. **`moves.py`** — parses each game's PGN, extracts every position where it was my move, and for each one records:
   - the FEN
   - the move I actually played
   - a short history of the 2 preceding board states (for short-term context)
   - a Stockfish (depth 12, run locally) centipawn evaluation of the position
3. **`position.py`** — encodes a board position into a `(51, 8, 8)` tensor: 17 planes for the current position (12 piece-type/color planes, 4 castling-rights planes, 1 en passant plane), × 3 for the current position plus 2 historical positions (zero-padded early in a game). Positions are always encoded from *my* perspective — "my pieces" and "their pieces," mirrored when I played black, rather than raw white/black.
4. **`build_dataset.py`** — turns `move_pairs.json` into `dataset.npz`: board tensors, and four label sets (from-square, to-square, promotion piece, and a Stockfish-derived value target normalized via `tanh(centipawn / 400)`).
5. **`trunk_layer.py`** — the model. Two conv layers (12→64→128 channels, ReLU, dropout) feed four output heads:
   - **from-square** (64-way classification)
   - **to-square** (64-way classification)
   - **promotion piece** (5-way: none/Q/R/B/N)
   - **value** (a single `tanh`-bounded position evaluation, trained against the Stockfish labels)
6. **`training_loop.py`** — trains all four heads jointly (summed cross-entropy + MSE loss), with early stopping and checkpointing on validation loss.
7. **`predict.py`** / **`bot_engine.py`** — move selection at inference time:
   - the policy heads produce the top-K candidate moves (keeps move *selection* anchored to my actual habits, not "objectively good" play)
   - a shallow minimax search (2–3 ply) evaluates the resulting positions using the value head, assuming the opponent plays their own top candidates well
   - a final `is_hanging()` check filters out any minimax pick that still leaves a piece hanging for a bad trade, falling back down the ranked list if needed
8. **`homemade.py`** (in a separate [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) instance) — wraps `predict_move()` as a Lichess "homemade" engine, so the bot can actually play live games via Lichess's Board API.

## Design decisions worth calling out

- **Policy picks candidates; value only judges outcomes.** The value head (trained on Stockfish's opinion of "good chess") is deliberately kept out of candidate *selection* — it only scores the consequences of moves the policy model would already consider. The intent is to catch blunders without drifting the bot's style toward generically strong play.
- **Promotions get their own classification head**, rather than assuming queen-always, because the training data showed real, deliberate underpromotions (a handful of knight promotions) worth preserving.
- **Board encoding is player-relative, not color-relative.** Positions are mirrored so "my pieces" always occupy the same planes regardless of whether I played white or black in that game — the network learns one unified "how I play," not two separate white/black styles.
- **History is zero-padded, not fabricated**, both early in real games and for hypothetical positions inside the search tree — repeating a stale position as fake "history" was judged more misleading than just signaling "no information here."

## Results (honest numbers, not cherry-picked)

On a held-out validation split (reproducible via a fixed `random_split` seed):

| Metric | Value |
|---|---|
| From-square accuracy | ~37–39% |
| To-square accuracy | ~24–26% |
| Exact full-move accuracy | ~13–14% |
| Value head MAE | 0.336 (vs. 0.639 for a naive mean-prediction baseline) |

Full-move accuracy is a strict metric — chess often has several reasonable replies in a given position, so a "miss" here doesn't necessarily mean a bad move, just a different one than I happened to play that day.

**What this project is not:** a strong engine. It has no deep calculation, and a shallow 2–3 ply minimax search — chosen deliberately, to keep decisions anchored to style rather than raw strength — will still lose to genuine multi-move tactics. That's by design, not an oversight.

## What was tried and didn't move the needle much

Documented here rather than left as silent dead ends:
- **Frame-stacking (2 prior positions)** — a small, borderline-noise-level accuracy bump. A real but modest lever; a static CNN over a few snapshots isn't a substitute for genuine sequence modeling.
- **Castling rights / en passant planes** — no measurable accuracy change, likely because these signals are low-frequency across ~24k positions. Kept anyway for completeness.
- **Dropout rate (0.3 vs 0.5)** — negligible difference in accuracy; both meaningfully reduce the val-loss blowup seen with no dropout at all.

## Stack

Python, PyTorch, `python-chess`, local Stockfish (position labeling only, not used at inference time), NumPy, `lichess-bot`.

## Possible next steps

- More training data at each future rating milestone (the actual highest-leverage lever available)
- A cleaner, sequence-aware architecture (RNN/Transformer over move history) if frame-stacking's limits become a real bottleneck
- Full search + self-play (AlphaZero-style) at a later milestone, explicitly as a *separate* mode from the style-cloning snapshots