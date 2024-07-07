import sys
from collections import defaultdict
from functools import lru_cache

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
    assert all(c in "#." for c in row), "# と . で構築してください"

# 通れるなら 1, 通れないなら 0 に変換
board = []
for h in range(H):
    row = []
    for w in range(W):
        row.append(1 if board_txt[h][w] == "." else 0)
    board.append(row)


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
directions = [((0, -1), (0, 1)), ((-1, 0), (1, 0))]  # ((左右), (上下))
cell_to_nodes = defaultdict(list)
node_to_cells = defaultdict(list)
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
for v, nodes in enumerate(condensed_graph):
    for u in nodes:
        condensed_G.add_edge(v, u)


# @debug
def can_fill_all_cells(nodes):
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


def can_absolutely_solve():
    """
    盤面が絶対に詰まないかどうかを判定する
    True なら絶対に詰まない
    False なら詰む可能性がある
    """
    # 縮約グラフ上での出次数が 0 の強連結成分に含まれるノードをすべてたどったとき、マスをすべて埋められるなら、その盤面は詰まない
    ret = True
    for v in range(len(condensed_graph)):
        # 出次数がゼロ
        if not condensed_graph[v]:
            scc_nodes = scc[v]
            ret &= can_fill_all_cells(scc_nodes)
    return ret


def main():
    print(f"絶対に詰まない: {can_absolutely_solve()}")

    # debug
    # # 図にする
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
