import random
import time

import solver

MAX_ITER = 30000
TIMEOUT = 60  # sec
ALLOW_CHECKMATE = True
INPUT_FROM_FILE = False

if INPUT_FROM_FILE:
    board_txt = open("_in.txt", "r").read().strip().split("\n")
    if "" in board_txt:
        board_txt = board_txt[:board_txt.index("")]
    H = len(board_txt)
    W = len(board_txt[0])
    for i, row in enumerate(board_txt):
        board_txt[i] = list(row)
else:
    H = 10
    W = 10
    board_txt = [['#'] * W for _ in range(H)]
    board_txt[H // 2][W // 2] = 'o'

start_time = time.time()
for i in range(MAX_ITER):
    idx = int(random.random() * (H * W))
    h = idx // W
    w = idx % W
    if board_txt[h][w] in 'o.':
        continue

    board_txt[h][w] = '.'
    start_cell, can_absolutely_solve, can_solve, solve_result = \
        solver.solve(board_txt, beam_search_beam_width=10000, beam_search_max_steps=1000, with_solve_result=False)
    # 解けるなら適用、解けないならもとに戻す
    if not can_solve:
        board_txt[h][w] = '#'
    # 詰む可能性があるならもとに戻す
    elif (not ALLOW_CHECKMATE) and (not can_absolutely_solve):
        board_txt[h][w] = '#'

    print(f"\r{i + 1}/{MAX_ITER}, {time.time() - start_time:.2f} sec")
    if i % 1000 == 0:
        print('\n'.join([''.join(row) for row in board_txt]))

    elapsed_time = time.time() - start_time
    if elapsed_time > TIMEOUT:
        break

print('\n'.join([''.join(row) for row in board_txt]))
start_cell, can_absolutely_solve, can_solve, solve_result = \
    solver.solve(board_txt, beam_search_beam_width=10000, beam_search_max_steps=1000)
print(f"スタート地点: {start_cell}")
print(f"絶対に詰まない？: {can_absolutely_solve}")
print(f"解が存在する？: {can_solve}")
if solve_result:
    steps, hist = solve_result
    print(f"最短手数 (BeamSearch): {steps}, 行動ログ: {hist}")
