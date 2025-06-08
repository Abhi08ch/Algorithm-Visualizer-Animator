import heapq
from typing import List, Tuple, Dict, Set, Optional
from typing import List
import copy

# Bubble Sort
def bubble_sort_steps(arr: List[int]):
    n = len(arr)
    arr = arr.copy()
    yield arr.copy()
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                yield arr.copy()

# Binary Search
def binary_search_steps(arr: List[int], target: int):
    left, right = 0, len(arr) - 1
    steps = []
    while left <= right:
        mid = (left + right) // 2
        steps.append({"array": arr.copy(), "left": left, "right": right, "mid": mid})
        if arr[mid] == target:
            break
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return steps

# Helper for grid neighbors
def neighbors(pos, grid, allow_diagonal=False):
    x, y = pos
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    if allow_diagonal:
        directions += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for dx, dy in directions:
        nx, ny = x+dx, y+dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
            yield (nx, ny)

# BFS for Pathfinding
def bfs_pathfinding_steps(grid: List[List[int]], start: Tuple[int,int], end: Tuple[int,int]):
    from collections import deque
    visited = [[False]*len(grid[0]) for _ in grid]
    parent = {}
    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True
    steps = []
    steps.append({"grid": [row[:] for row in grid], "visited": [row[:] for row in visited], "frontier": [start], "path": []})
    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            steps.append({"grid": [row[:] for row in grid], "visited": [row[:] for row in visited], "frontier": [], "path": path})
            return steps
        for nbr in neighbors(current, grid):
            if not visited[nbr[0]][nbr[1]]:
                visited[nbr[0]][nbr[1]] = True
                parent[nbr] = current
                queue.append(nbr)
                steps.append({"grid": [row[:] for row in grid], "visited": [row[:] for row in visited], "frontier": [nbr], "path": []})
    steps.append({"grid": [row[:] for row in grid], "visited": [row[:] for row in visited], "frontier": [], "path": []})
    return steps

# DFS for Pathfinding
def dfs_pathfinding_steps(grid: List[List[int]], start: Tuple[int,int], end: Tuple[int,int]):
    steps = []
    stack = [start]
    visited = set()
    parent = {}
    while stack:
        node = stack.pop()
        if node == end:
            break
        if node not in visited:
            visited.add(node)
            for nbr in neighbors(node, grid):
                if nbr not in visited:
                    stack.append(nbr)
                    parent[nbr] = node
            steps.append({
                "grid": [row[:] for row in grid],
                "visited": list(visited),
                "frontier": stack.copy(),
                "path": []
            })
    # reconstruct path
    path = []
    n = end
    while n in parent:
        path.append(n)
        n = parent[n]
    path.append(start)
    path.reverse()
    steps.append({
        "grid": [row[:] for row in grid],
        "visited": list(visited),
        "frontier": [],
        "path": path
    })
    return steps

# Dijkstra's Algorithm
def dijkstra_pathfinding_steps(grid: List[List[int]], start: Tuple[int,int], end: Tuple[int,int]):
    heap = [(0, start)]
    distances = {start: 0}
    parent = {}
    visited = set()
    steps = []
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        steps.append({"grid": [row[:] for row in grid], "visited": list(visited), "frontier": [node], "path": []})
        if node == end:
            break
        for nbr in neighbors(node, grid):
            if nbr not in visited:
                new_dist = dist + 1
                if new_dist < distances.get(nbr, float('inf')):
                    distances[nbr] = new_dist
                    parent[nbr] = node
                    heapq.heappush(heap, (new_dist, nbr))
    # reconstruct path
    path = []
    n = end
    while n in parent:
        path.append(n)
        n = parent[n]
    path.append(start)
    path.reverse()
    steps.append({"grid": [row[:] for row in grid], "visited": list(visited), "frontier": [], "path": path})
    return steps

# Prim's Algorithm (for MST)
def prim_steps(graph: Dict[int, List[Tuple[int, int]]], start: int):
    import heapq
    visited = set()
    mst = []
    heap = []
    steps = []
    for v, w in graph[start]:
        heapq.heappush(heap, (w, start, v))
    visited.add(start)
    steps.append({"visited": set(visited), "mst": list(mst), "heap": heap.copy()})
    while heap:
        w, u, v = heapq.heappop(heap)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, w))
            for to, weight in graph[v]:
                if to not in visited:
                    heapq.heappush(heap, (weight, v, to))
            steps.append({"visited": set(visited), "mst": list(mst), "heap": heap.copy()})
    return steps

# Kruskal's Algorithm (for MST)
def kruskal_steps(edges: List[Tuple[int,int,int]], n_nodes: int):
    parent = list(range(n_nodes))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        parent[find(u)] = find(v)
    mst = []
    edges = sorted(edges, key=lambda x: x[2])
    steps = []
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))
        steps.append({"mst": list(mst), "parent": list(parent)})
    return steps

# BST insert and traversals
class BSTNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def bst_insert_steps(values: List[int]):
    if not values:
        return []
    root = BSTNode(values[0])
    steps = [{"tree": bst_to_list(root), "root": copy.deepcopy(root)}]
    for v in values[1:]:
        node = root
        while True:
            if v < node.val:
                if node.left:
                    node = node.left
                else:
                    node.left = BSTNode(v)
                    break
            else:
                if node.right:
                    node = node.right
                else:
                    node.right = BSTNode(v)
                    break
        # The fix: deepcopy root so each step is an independent snapshot
        steps.append({"tree": bst_to_list(root), "root": copy.deepcopy(root)})
    return steps

def bst_to_list(root):
    res = []
    queue = [root]
    while queue:
        curr = queue.pop(0)
        if curr:
            res.append(curr.val)
            queue.append(curr.left)
            queue.append(curr.right)
        else:
            res.append(None)
    while res and res[-1] is None:
        res.pop()
    return res