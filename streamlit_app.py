import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import solver

# カラーマップの定義
# 0: まだ塗ってない
# 1: 塗った
# 2: 壁
color_map = plt.cm.colors.ListedColormap(['white', 'red', 'black'])
COLOR_WHITE = 0
COLOR_RED = 1
COLOR_BLACK = 2

st.header("AMAZE Solver")
st.write("盤面を指定してください")
with st.expander("サンプル"):
    st.subheader('解けるパターン')
    st.code('''
#....
.....
..o..
.....
....#
    ''', language="text")
    st.code('''
o.#...#
.......
.......
.....#.
.#.....
.......
......#
    ''', language="text")
    # st.write('全体が 1 つの SCC のパターン (詰まない)')
    st.code('''
.............
o..........#.
##########.#.
#########..#.
........#..#.
..#.....#..#.
..#.#####.##.
..#..........
..#####......
#.#####.#.#.#
....###.#.#.#
..#.###.#.#.#
..#.....#...#
    ''', language="text")

    st.subheader('詰む可能性があるが、解けるパターン')
    st.code('''
...##
##.#.
.#.#.
..o..
####.
    ''', language="text")
    st.code('''
...##
##.##
.#.##
..o..
#####
    ''', language="text")
    # st.write('出次数 0 の SCC が 1 つあるパターン (詰むことがあるが解ける)')
    st.code('''
#o.##
....#
###.#
.....
    ''', language="text")

    st.subheader('解けないパターン')
    st.code('''
...##
##.#.
.#.#.
....o
####.
    ''', language="text")
    # st.write('出次数 0 の SCC が 2 つあるパターン (詰む、解けない)')
    st.code('''
o..##
##.##
.#.#.
.....
.###.
    ''', language="text")

sample_stage = """
#....
.....
..o..
.....
....#
""".strip()
st.text_area('"#":壁, ".":地面, "o":開始位置', sample_stage, height=200, key="board_txt")

# 盤面を構築
board_txt: List[str] = st.session_state.board_txt.strip().split("\n")

H = len(board_txt)
W = len(board_txt[0])
for row in board_txt:
    assert len(row) == W, f"横を揃えてください: {row}"
    assert all(c in "#.o" for c in row), "# と . と o で構築してください"

# st.container() は場所だけ
# 描画場所
grid_container = st.empty()
# ステップ数表示場所
step_count_container = st.empty()
# 再生ボタン
play_button = st.button("再生")
# エラー表示場所
error_container = st.empty()
table_container = st.empty()

# スライダーで円の移動速度を調整
speed = st.sidebar.slider("Speed", 1, 60, 60)
# ビーム幅
beam_widths = [1000, 3000, 10000, 30000, 100000]
beam_width = st.sidebar.selectbox("Beam Width", beam_widths, index=len(beam_widths) - 1)

# 再生中に途中の動きを見せるか
animation_enabled = False
if os.getenv("LOCAL"):
    animation_enabled = st.sidebar.checkbox("Animate", value=True)


def plot_state(colors, h, w):
    """
    現在の状態をプロットする
    """
    # セルの中央に移す
    x = w + 0.5
    # セルの中央に移す・上下反転
    y = H - h - 0.5
    circle_radius = 70 / max(H, W)
    circle_outline = circle_radius / 15
    circle_s = circle_radius ** 2

    # グリッドと円をプロット
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(colors, cmap=color_map, extent=(0, W, 0, H), vmin=0, vmax=2)

    ax.scatter(x, y, c='red', edgecolor='black', s=circle_s, linewidth=circle_outline)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.tick_params(bottom=False, left=False, right=False, top=False)
    # グリッド線を描画
    ax.set_xticks(np.arange(0, W + 1, 1))
    ax.set_yticks(np.arange(0, H + 1, 1))
    ax.grid(color='black', linestyle='-', linewidth=0.5)
    # Streamlitで画像を更新
    grid_container.pyplot(fig, use_container_width=False)
    # プロットを閉じる
    plt.close(fig)


# グリッドを白黒赤で塗り分ける
colors = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        colors[i, j] = COLOR_WHITE if board_txt[i][j] == "." else COLOR_RED if board_txt[i][j] == "o" else COLOR_BLACK
start_cell = np.where(colors == COLOR_RED)
plot_state(colors, *start_cell)


# キャッシュ
@st.cache_data(max_entries=1)
def solve(board_txt, beam_width):
    progress = st.progress(0.0)

    def update_progress(step, rate):
        progress.progress(rate)

    ret = solver.solve(
        board_txt,
        beam_search_beam_width=beam_width,
        beam_search_max_steps=1000,
        beam_search_step_callback=update_progress)
    progress.progress(1.0)
    progress.empty()
    return ret


start_cell, can_absolutely_solve, can_solve, solve_result = solve(board_txt, beam_width)

min_steps = solve_result[0] if solve_result else -1
hist = solve_result[1] if solve_result else []

if not start_cell:
    error_container.error('スタート地点を "o" で指定してください')
elif not can_solve:
    error_container.error('解が存在しません')

directions = np.sign(np.array(list(hist[1:])) - np.array(list(hist[:-1])))
operations = list(map(lambda d: '↑' if d[0] == -1 else '↓' if d[0] == 1 else '→' if d[1] == 1 else '←', directions))
table_container.table({
    "解ける？": 'Yes' if can_solve else 'No',
    "詰む可能性がある？": 'No' if can_absolutely_solve else 'Yes',
    "最短手数": min_steps,
    "最短操作ログ": ''.join(operations)
})


def is_valid(h, w):
    return 0 <= h < H and 0 <= w < W


step_count_container.write(f"{0}/{len(hist) - 1}")
if can_solve and play_button:
    for step in range(1, len(hist)):
        prev = hist[step - 1]
        current = hist[step]

        step_count_container.write(f"{operations[step - 1]} {step}/{len(hist) - 1}")

        # 4 方向前提になってる
        dh, dw = np.sign(np.array(list(current)) - np.array(list(prev)))
        h, w = prev
        while (h, w) != current:
            colors[h, w] = COLOR_RED
            if animation_enabled:
                plot_state(colors, h, w)
                time.sleep(1 / speed)
            h += dh
            w += dw
        colors[*current] = COLOR_RED

        plot_state(colors, *current)
        time.sleep(3 / speed)

    st.balloons()
