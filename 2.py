import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from time import perf_counter

g = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': ['F'], 'F': []
}

def search(start, goal, bfs=True):
    q = deque([(start, [start])])
    visited = set()
    order = []
    t = perf_counter()

    while q:
        node, path = q.popleft() if bfs else q.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        if node == goal:
            return path, order, perf_counter() - t

        for n in g[node]:
            if n not in visited:
                q.append((n, path + [n]))

    return [], order, perf_counter() - t

for name, flag in [("BFS", True), ("DFS", False)]:
    path, order, duration = search('A', 'F', flag)
    print(name, "Path:", path)
    print("Traversal:", order)
    print("Nodes explored:", len(order))
    print("Path length:", max(0, len(path)-1))
    print("Time:", duration, "seconds")

G = nx.DiGraph(g)
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=1200, arrows=True)
plt.title("BFS and DFS Search Graph")
plt.show()