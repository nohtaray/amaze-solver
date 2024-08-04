import solver

# _in.txt から盤面を構築
board_txt = open("_in.txt", "r").read().strip().split("\n")
# 空行以降は読み込まない
if "" in board_txt:
    board_txt = board_txt[:board_txt.index("")]

start_cell, can_absolutely_solve, can_solve, solve_result = \
    solver.solve(board_txt, beam_search_beam_width=10000, beam_search_max_steps=1000)

if start_cell:
    print(f"スタート地点: {start_cell}")
    print(f"絶対に詰まない？: {can_absolutely_solve}")
    print(f"解が存在する？: {can_solve}")
    if solve_result:
        steps, hist = solve_result
        print(f"最短手数 (BeamSearch): {steps}, 行動ログ: {hist}")
else:
    print("スタート地点: 任意")
    print(f"絶対に詰まない？: {can_absolutely_solve}")
