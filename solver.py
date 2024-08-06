import heapq
import os
import sys
from collections import defaultdict
from functools import lru_cache
from typing import DefaultDict, List, Tuple, Set

import matplotlib.pyplot as plt
import networkx as nx

from debug import debug


def solve(board_txt: List[str],
          beam_search_beam_width=1000,
          beam_search_max_steps=1000,
          beam_search_step_callback=None,
          with_solve_result=True):
    sys.setrecursionlimit(10 ** 9)

    H = len(board_txt)
    W = len(board_txt[0])
    for row in board_txt:
        assert len(row) == W, f"横を揃えてください: {row}"
        assert all(c in "#.o" for c in row), "# と . と o で構築してください"

    # スタート地点を探す
    start_cell = None
    for h in range(H):
        for w in range(W):
            if board_txt[h][w] == 'o':
                if start_cell:
                    raise ValueError("スタート地点は 1 つだけにしてください")
                start_cell = h, w

    # 通れる場所を列挙
    all_cells = set()
    for h in range(H):
        for w in range(W):
            if board_txt[h][w] != "#":
                all_cells.add((h, w))

    # 移動方向の候補
    # 斜めもあるならここに定義
    directions = [((0, -1), (0, 1)), ((-1, 0), (1, 0))]  # ((左右), (上下))
    directions_flatten = [d for dirs in directions for d in dirs]

    # 何マス塗ったらクリアか
    clear_draw_count = len(all_cells)

    # あるマスから、その方向にまっすぐいったときにどこで止まるかを返す
    @lru_cache(maxsize=None)
    # @debug
    def get_end(h, w, direction):
        if (h, w) not in all_cells:
            return h, w
        dh, dw = direction
        if (h + dh, w + dw) in all_cells:
            return get_end(h + dh, w + dw, direction)
        return h, w

    # @debug
    def get_cells_in_direction(h, w, direction):
        """
        あるマスから、その方向にまっすぐいったときに通るマスのリストを返す
        """
        ret = []
        dh, dw = direction
        while (h, w) in all_cells:
            ret.append((h, w))
            h += dh
            w += dw
        return ret

    # 直線移動の始点と終点のセットをノードにする
    # あるマスから移動できるノードを列挙する
    cell_to_nodes: DefaultDict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
    node_to_cells: DefaultDict[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[int, int]]] = defaultdict(list)
    for h, w in all_cells:
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
        visited = set()
        for node in nodes:
            for h, w in node_to_cells[node]:
                visited.add((h, w))
        return len(all_cells & visited) == len(all_cells)

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
        visited = set()
        visited.add((start_h, start_w))
        stack = [start_cell]
        while stack:
            cell = stack.pop()
            if cell in goal_cells:
                return True

            for dh, dw in directions_flatten:
                h, w = cell
                # まっすぐ行けるとこまで行く
                while (h, w) != forbidden_cell and (h + dh, w + dw) in all_cells:
                    h += dh
                    w += dw
                # 禁止マスを通らずに止まれたらスタックに積む
                if (h, w) != forbidden_cell and (h, w) not in visited:
                    visited.add((h, w))
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
        for h, w in all_cells:
            if (h, w) not in last_scc_cells:
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
            for (h, w) in all_cells:
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

    def hash_canvas(canvas):
        return hash(tuple(map(tuple, canvas)))

    class CanvasState:
        def __init__(self, canvas):
            self.canvas = canvas
            self.hash = hash_canvas(canvas)
            self.draw_count = sum(sum(row) for row in canvas)
            self.is_clear = self.draw_count == clear_draw_count

        def __hash__(self):
            return self.hash

        def __eq__(self, other):
            return self.hash == other.hash

        def __lt__(self, other):
            return self.draw_count < other.draw_count

        def get_canvas_copy(self):
            return [list(row) for row in self.canvas]

    hash_to_canvas = dict()

    def get_canvas_state(canvas):
        hash = hash_canvas(canvas)
        if hash not in hash_to_canvas:
            hash_to_canvas[hash] = CanvasState([list(row) for row in canvas])
        return hash_to_canvas[hash]

    # 最短手数を求める
    # @debug
    def minimum_steps__beam_search(start_cell, beam_width=1000, max_steps=1000, step_callback=None):
        start_h, start_w = start_cell
        initial_canvas = [[0] * W for _ in range(H)]
        initial_canvas[start_h][start_w] = 1
        # ビームサーチ
        initial_key = get_canvas_state(initial_canvas), start_cell
        beam = [initial_key]
        seen = {initial_key}
        prev = {initial_key: None}

        def restore_hist(key):
            hist = []
            while key:
                _, cell = key
                hist.append(cell)
                key = prev[key]
            return hist[::-1]

        steps = 0
        while beam and steps < max_steps:
            next_beam = set()
            for canvas_state, (h, w) in beam:
                for direction in directions_flatten:
                    draw_cells = get_cells_in_direction(h, w, direction)
                    canvas = canvas_state.get_canvas_copy()
                    for draw_h, draw_w in draw_cells:
                        canvas[draw_h][draw_w] = 1
                    next_canvas_state = get_canvas_state(canvas)
                    if next_canvas_state.is_clear:
                        return steps + 1, restore_hist((canvas_state, (h, w))) + [draw_cells[-1]]
                    key = (next_canvas_state, draw_cells[-1])
                    if key not in seen:
                        next_beam.add(key)
                        prev[key] = canvas_state, (h, w)
            # 全部見た
            if not next_beam:
                return -1, []

            # 進捗がいいものを beam_width だけ残す
            beam = sorted(next_beam, key=lambda x: -x[0].draw_count)[:beam_width]
            seen.update(beam)
            steps += 1

            # debug
            progress = beam[0][0].draw_count / clear_draw_count
            if os.getenv("DEBUG"):
                print(
                    f"step: {steps}, progress: {progress:.2f} ({beam[0][0].draw_count}/{clear_draw_count}), seen: {len(seen)}")
            if step_callback:
                step_callback(steps, progress)
        return -1, []

    # 最短手数を求める
    @debug
    def minimum_steps__a_star(start_cell, max_iter=100000):
        start_h, start_w = start_cell
        initial_canvas = [[0] * W for _ in range(H)]
        initial_canvas[start_h][start_w] = 1
        initial_key = get_canvas_state(initial_canvas), start_cell
        # 評価関数
        # draw_count はおおきいほどよい、step は小さいほどよい
        evaluate = lambda draw_count, step: -draw_count + step

        heap = [(evaluate(1, 0), 0, initial_key)]
        seen = {initial_key: 0}
        prev = {initial_key: None}

        def restore_hist(key):
            hist = []
            while key:
                _, cell = key
                hist.append(cell)
                key = prev[key]
            return hist[::-1]

        optimal = (float('inf'), -1, [])
        idx = 0
        while heap and idx < max_iter:
            _, step, (canvas_state, (h, w)) = heapq.heappop(heap)
            for direction in directions_flatten:
                draw_cells = get_cells_in_direction(h, w, direction)
                canvas = canvas_state.get_canvas_copy()
                for draw_h, draw_w in draw_cells:
                    canvas[draw_h][draw_w] = 1
                next_canvas_state = get_canvas_state(canvas)
                key = (next_canvas_state, draw_cells[-1])
                if next_canvas_state.is_clear:
                    val = evaluate(next_canvas_state.draw_count, step + 1)
                    optimal_val, *_ = optimal
                    if val < optimal_val:
                        optimal = val, step + 1, key
                if key not in seen or seen[key] > step + 1:
                    heapq.heappush(heap, (evaluate(next_canvas_state.draw_count, step + 1), step + 1, key))
                    seen[key] = step + 1
                    prev[key] = canvas_state, (h, w)

            idx += 1

            # debug
            if idx % 10000 == 0:
                print(
                    f"idx: {idx}, seen: {len(seen)}, heap: {len(heap)}, optimal: {optimal[0]}, step: {optimal[1]}")

        if optimal[1] != -1:
            return optimal[1], restore_hist(optimal[2])
        return -1, []

    # 最短手数を求める
    # @debug
    def minimum_steps__chokudai_search(start_cell, max_beam_width=1000, max_steps=100):
        start_h, start_w = start_cell
        initial_canvas = [[0] * W for _ in range(H)]
        initial_canvas[start_h][start_w] = 1
        initial_key = get_canvas_state(initial_canvas), start_cell
        # 評価関数
        # draw_count はおおきいほどよい、step は小さいほどよい
        evaluate = lambda draw_count, step: -draw_count + step

        optimal = (float('inf'), -1, [])
        prev = {initial_key: None}

        def restore_hist(key):
            hist = []
            while key:
                _, cell = key
                hist.append(cell)
                key = prev[key]
            return hist[::-1]

        step_heaps = [[] for _ in range(max_steps)]
        step_heaps[0].append((evaluate(1, 0), 0, initial_key))
        for beam_width in range(1, max_beam_width + 1):
            for step in range(max_steps - 1):
                if not step_heaps[step]:
                    continue
                _, _, (canvas_state, (h, w)) = heapq.heappop(step_heaps[step])
                for direction in directions_flatten:
                    draw_cells = get_cells_in_direction(h, w, direction)
                    canvas = canvas_state.get_canvas_copy()
                    for draw_h, draw_w in draw_cells:
                        canvas[draw_h][draw_w] = 1
                    next_canvas_state = get_canvas_state(canvas)
                    key = (next_canvas_state, draw_cells[-1])
                    if next_canvas_state.is_clear:
                        val = evaluate(next_canvas_state.draw_count, step + 1)
                        optimal_val, *_ = optimal
                        if val < optimal_val:
                            optimal = val, step + 1, key
                    else:
                        # FIXME: 入れ過ぎたら削除していかないといけない気がする
                        heapq.heappush(step_heaps[step + 1],
                                       (evaluate(next_canvas_state.draw_count, step + 1), step + 1, key))
                        prev[key] = canvas_state, (h, w)

            # debug
            if beam_width % 100 == 0:
                print(f"beam_width: {beam_width}, optimal: {optimal}")

        if optimal[1] != -1:
            return optimal[1], restore_hist(optimal[2])
        return -1, []

    def main():
        ret = {}
        ret["start_cell"] = start_cell
        if start_cell:
            ret["can_absolutely_solve"] = can_absolutely_solve_from_cell(start_cell)
            ret["can_solve"] = can_solve_from_cell(start_cell)
            if with_solve_result and ret["can_solve"]:
                steps, hist = minimum_steps__beam_search(
                    start_cell,
                    beam_width=beam_search_beam_width,
                    max_steps=beam_search_max_steps,
                    step_callback=beam_search_step_callback)
                # print(f"最短手数 (BeamSearch): {steps}, 行動ログ: {hist}")
                # steps, hist = minimum_steps__a_star(start_cell)
                # print(f"最短手数 (A*): {steps}, {hist}")
                # steps, hist = minimum_steps__chokudai_search(start_cell)
                # print(f"最短手数 (ChokudaiSearch): {steps}, {hist}")
                ret["solve_result"] = steps, hist
        else:
            # スタート地点が指定されていない場合は、絶対に詰まないかどうかだけを返す
            ret["can_absolutely_solve"] = can_absolutely_solve()
            ret["can_solve"] = can_absolutely_solve()
        return ret.get("start_cell"), ret.get("can_absolutely_solve"), ret.get("can_solve"), ret.get("solve_result")

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

    return main()
