import sys
from collections import defaultdict
from functools import lru_cache
from typing import DefaultDict, List, Tuple, Set

import matplotlib.pyplot as plt
import networkx as nx

from debug import debug

sys.setrecursionlimit(10 ** 9)

# _in.txt から盤面を構築
board_txt = open("_in.txt", "r").read().strip().split("\n")
# 空行以降は読み込まない
if "" in board_txt:
    board_txt = board_txt[:board_txt.index("")]

H = len(board_txt)
W = len(board_txt[0])
for row in board_txt:
    assert len(row) == W, f"横を揃えてください: {row}"
    assert all(c in "#.o" for c in row), "# と . と o で構築してください"

# スタート地点が書いてあればそれを探す
start_cell = None
for h in range(H):
    for w in range(W):
        if board_txt[h][w] == 'o':
            if start_cell:
                raise ValueError("スタート地点が複数あります")
            start_cell = h, w

# 通れるなら 1, 通れないなら 0 に変換
board = []
for h in range(H):
    row = []
    for w in range(W):
        row.append(1 if board_txt[h][w] != "#" else 0)
    board.append(row)

# 移動方向の候補
# 斜めもあるならここに定義
directions = [((0, -1), (0, 1)), ((-1, 0), (1, 0))]  # ((左右), (上下))
directions_flatten = [d for dirs in directions for d in dirs]


def is_all_connected():
    """
    盤面が連結かどうか
    """
    visited = [[False] * W for _ in range(H)]
    stack = []
    for h in range(H):
        for w in range(W):
            if board[h][w]:
                stack.append((h, w))
                break
        if stack:
            break

    while stack:
        h, w = stack.pop()
        for dh, dw in directions_flatten:
            nh, nw = h + dh, w + dw
            if 0 <= nh < H and 0 <= nw < W and board[nh][nw] and not visited[nh][nw]:
                visited[nh][nw] = True
                stack.append((nh, nw))

    for h in range(H):
        for w in range(W):
            if board[h][w] and not visited[h][w]:
                return False
    return True


if not is_all_connected():
    raise ValueError("盤面に飛び地があります")


# あるマスから、その方向にまっすぐいったときにどこで止まるかを返す
@lru_cache(maxsize=None)
# @debug
def get_end(h, w, direction):
    if not board[h][w]:
        return h, w
    dh, dw = direction
    if 0 <= h + dh < H and 0 <= w + dw < W and board[h + dh][w + dw]:
        return get_end(h + dh, w + dw, direction)
    return h, w


# 直線移動の始点と終点のセットをノードにする
# あるマスから移動できるノードを列挙する
cell_to_nodes: DefaultDict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
node_to_cells: DefaultDict[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[int, int]]] = defaultdict(list)
for h in range(H):
    for w in range(W):
        if not board[h][w]:
            continue
        cell = h, w
        for dir1, dir2 in directions:
            node = get_end(h, w, dir1), get_end(h, w, dir2)
            cell_to_nodes[cell].append(node)
            node_to_cells[node].append(cell)

# ある直線から遷移できる直線を結ぶグラフを作る
graph = dict()
for nodes in cell_to_nodes.values():
    for end1, end2 in nodes:
        if (end1, end2) in graph:
            continue
        graph[(end1, end2)] = []
        for node in cell_to_nodes[end1]:
            if node != (end1, end2):
                graph[(end1, end2)].append(node)
        for node in cell_to_nodes[end2]:
            if node != (end1, end2):
                graph[(end1, end2)].append(node)
G = nx.DiGraph()
G.add_nodes_from(graph.keys())
for k, v in graph.items():
    for vv in v:
        G.add_edge(k, vv)

# 強連結成分分解
scc = list(nx.strongly_connected_components(G))
# ノードから属している強連結成分を取得するマップ
node_to_scc = dict()
for i, nodes in enumerate(scc):
    for node in nodes:
        node_to_scc[node] = i

# 縮約グラフ
condensed_graph = [set() for _ in range(len(scc))]
for v, nodes in enumerate(scc):
    for node in nodes:
        for u in G[node]:
            if node_to_scc[u] != v:
                condensed_graph[v].add(node_to_scc[u])
condensed_G = nx.DiGraph()
condensed_G.add_nodes_from(range(len(condensed_graph)))
for v, nodes in enumerate(condensed_graph):
    for u in nodes:
        condensed_G.add_edge(v, u)


# @debug
def can_fill_all_cells(nodes: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    """
    指定したノードですべてのマスを埋められるか
    """
    visited = [[False] * W for _ in range(H)]
    for node in nodes:
        for h, w in node_to_cells[node]:
            visited[h][w] = True
    ok = True
    for h in range(H):
        for w in range(W):
            if board[h][w]:
                ok &= visited[h][w]
    return ok


@debug
def can_reach_to(goal_cells: Set[Tuple[int, int]], start_cell, forbidden_cell):
    """
    指定した開始位置から指定したマスのどれかに forbidden_cell を通らずに到達できるか
    """
    if start_cell == forbidden_cell:
        return False
    if start_cell in goal_cells:
        return True

    start_h, start_w = start_cell
    visited = [[False] * W for _ in range(H)]
    visited[start_h][start_w] = True
    stack = [start_cell]
    while stack:
        cell = stack.pop()
        if cell in goal_cells:
            return True

        for dh, dw in directions_flatten:
            h, w = cell
            # まっすぐ行けるとこまで行く
            while (h, w) != forbidden_cell and 0 <= h + dh < H and 0 <= w + dw < W and board[h + dh][w + dw]:
                h += dh
                w += dw
            # 禁止マスを通らずに止まれたらスタックに積む
            if (h, w) != forbidden_cell and not visited[h][w]:
                visited[h][w] = True
                stack.append((h, w))
    return False


@lru_cache(maxsize=None)
def scc_to_cells(scc_v):
    """
    強連結成分の番号からその強連結成分に含まれるマスのセットを返す
    """
    cells = set()
    for node in scc[scc_v]:
        for h, w in node_to_cells[node]:
            cells.add((h, w))
    return cells


@debug
def can_absolutely_fill_all_cells_from_cell(start_cell, scc_v):
    """
    指定した開始位置から scc_v の連結成分に向かうパスのすべてで、すべてのマスを埋められるか
    """
    # これらのマスに来ると最後の連結成分に入る
    last_scc_cells = scc_to_cells(scc_v)

    # 埋めないといけないマス
    not_visited_cells = []
    for h in range(H):
        for w in range(W):
            if board[h][w] and (h, w) not in last_scc_cells:
                not_visited_cells.append((h, w))

    # 埋めないといけないマスを通らずに到達できるパスがあるなら、すべてのマスを埋めることはできない
    for forbidden_cell in not_visited_cells:
        if can_reach_to(last_scc_cells, start_cell, forbidden_cell):
            return False
    return True


@debug
def can_absolutely_solve():
    """
    盤面のどこからスタートしても絶対に詰まないかどうかを判定する
    True なら絶対に詰まない
    False なら詰む可能性がある (解ける可能性もある)
    """
    # 縮約グラフ上での出次数が 0 の強連結成分に含まれるノードをすべてたどったとき、マスをすべて埋められるなら、その盤面は詰まない
    for v in range(len(condensed_graph)):
        # 出次数がゼロ
        if not condensed_graph[v]:
            scc_nodes = scc[v]
            if not can_fill_all_cells(scc_nodes):
                return False
    return True


@debug
def can_absolutely_solve_from_cell(start_cell):
    """
    指定した位置から開始して、絶対に詰まないかどうかを判定する
    True なら絶対に詰まない
    False なら詰む可能性がある (解ける可能性もある)
    """
    # どこから開始しても絶対に詰まないなら True でいい
    if can_absolutely_solve():
        return True

    # 縮約グラフ上での出次数が 0 の強連結成分のいずれかについて、
    # スタート地点からそこに向かうパスのうち盤面を埋められないものがあれば、詰む可能性がある
    for v in range(len(condensed_graph)):
        # 出次数がゼロ
        if not condensed_graph[v]:
            if not can_absolutely_fill_all_cells_from_cell(start_cell, v):
                return False
    return True


def scc_dfs(scc_v, visit_counts=None):
    """
    強連結成分 scc_v から始めて、すべてのマスを埋められるパスがあるか
    """
    if not visit_counts:
        visit_counts = [[0] * W for _ in range(H)]
        for h, w in scc_to_cells(scc_v):
            visit_counts[h][w] += 1

    # 先端までいったら全部埋まってるか判定
    if not condensed_graph[scc_v]:
        ok = True
        for h in range(H):
            for w in range(W):
                if board[h][w]:
                    ok &= visit_counts[h][w] > 0
        return ok
    else:
        # 先があるなら続ける
        for scc_u in condensed_graph[scc_v]:
            for h, w in scc_to_cells(scc_u):
                visit_counts[h][w] += 1
            ok = scc_dfs(scc_u, visit_counts)
            for h, w in scc_to_cells(scc_u):
                visit_counts[h][w] -= 1

            if ok:
                return True
    return False


@debug
def can_solve_from_cell(start_cell):
    """
    指定した位置から開始して、解き方があるかどうかを判定する
    True なら解き方がある (詰む可能性もある)
    False なら絶対に解けない
    """
    # 絶対に解けるなら調べる必要ない
    if can_absolutely_solve_from_cell(start_cell):
        return True

    seen_scc_v = set()
    for start_node in cell_to_nodes[start_cell]:
        scc_v = node_to_scc[start_node]
        # すでにやったならスキップ
        if scc_v in seen_scc_v:
            continue
        seen_scc_v.add(scc_v)

        # scc_v から始まるパスのいずれかで、すべてのマスを埋められるものがあれば、解ける
        if scc_dfs(scc_v):
            return True
    return False


def main():
    if start_cell:
        print(f"スタート地点: {start_cell}")
        print(f"絶対に詰まない？: {can_absolutely_solve_from_cell(start_cell)}")
        print(f"解が存在する？: {can_solve_from_cell(start_cell)}")
    else:
        print("スタート地点: 任意")
        print(f"絶対に詰まない？: {can_absolutely_solve()}")

    # debug
    # 図にする
    # nx.draw_networkx(G, pos=nx.spring_layout(G, k=0.7), with_labels=True, arrows=True)
    # plt.show()
    # nx.draw_networkx(condensed_G, pos=nx.spring_layout(condensed_G, k=0.7), with_labels=True, arrows=True)
    # plt.show()
    # for v, nodes in enumerate(scc):
    #     print(f"強連結成分 {v}: {nodes}")
    #     for node in nodes:
    #         print(f"  {node} -> {node_to_cells[node]}")
    #     print()


if __name__ == "__main__":
    main()
