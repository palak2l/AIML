import heapq
import networkx as nx
import matplotlib.pyplot as plt

g = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('E', 1)],
    'D': [('F', 3)],
    'E': [('F', 2)],
    'F': []
}

h = {'A': 5, 'B': 4, 'C': 2,
     'D': 2, 'E': 1, 'F': 0}

def search(start, goal, astar=False):
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    visited = set()

    while pq:
        _, cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost, len(visited)

        for nxt, w in g[node]:
            new_cost = cost + w
            if nxt not in visited:
                if not astar or new_cost < best.get(nxt, float('inf')):
                    best[nxt] = new_cost
                    f = new_cost + h[nxt] if astar else h[nxt]
                    heapq.heappush(pq, (f, new_cost, nxt, path+[nxt]))

    return [], float('inf'), len(visited)

G = nx.DiGraph()
for u in g:
    for v, w in g[u]:
        G.add_edge(u, v, weight=w)

pos = nx.spring_layout(G, seed=42)

for name, flag in [("Greedy", False), ("A*", True)]:
    path, cost, explored = search('A', 'F', flag)
    print(name, "Path:", path, "Cost:", cost)
    print("Nodes explored:", explored)

    edges = list(zip(path, path[1:]))
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=1000)
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color="red", width=3)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title(name + " Search Path")
    plt.show()