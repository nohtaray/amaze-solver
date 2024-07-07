import bisect
import cmath
import heapq
import itertools
import math
import operator
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict, deque
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache, reduce
from math import gcd
from operator import add, itemgetter, mul, xor

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.setrecursionlimit(10 ** 9)

# _in.txt から盤面を構築
board_txt = open("_in.txt", "r").read().strip().split("\n")
# 空行で終わる
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

# カドを適当にスタート地点にする
start_h, start_w = -1, -1
for h in range(H):
    for w in range(W):
        if board[h][w]:
            start_h, start_w = h, w
            break
    if start_h >= 0:
        break


# あるマスから、その方向にまっすぐいったときにどこで止まるかを返す
@lru_cache(maxsize=None)
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
cell_to_nodes = dict()
for h in range(H):
    for w in range(W):
        if not board[h][w]:
            continue
        cell_to_nodes[(h, w)] = []
        for dir1, dir2 in directions:
            cell_to_nodes[(h, w)].append((get_end(h, w, dir1), get_end(h, w, dir2)))

# グラフにする
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

# 図にする
G = nx.DiGraph()
for k, v in graph.items():
    for vv in v:
        G.add_edge(k, vv)
nx.draw_networkx(G, pos=nx.spring_layout(G, k=0.7), with_labels=True, arrows=True)
plt.show()
