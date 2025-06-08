import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import networkx as nx
import time

def visualize_array(arr, step, pred=None):
    fig, ax = plt.subplots()
    indices = np.arange(len(arr))
    ax.bar(indices, arr, color='deepskyblue', alpha=0.9, label='Current')
    if pred is not None:
        ax.bar(indices, pred, color='orange', alpha=0.4, label='Predicted')
    ax.set_title(f"Step {step}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

def visualize_binary_search(arr, left, right, mid):
    fig, ax = plt.subplots()
    indices = np.arange(len(arr))
    colors = ['#ccc'] * len(arr)
    for i in range(left, right+1):
        colors[i] = 'deepskyblue'
    colors[mid] = 'red'
    ax.bar(indices, arr, color=colors)
    ax.set_title(f"Binary Search: left={left}, right={right}, mid={mid}")
    st.pyplot(fig)

def visualize_grid_step(step_data, start, end, algo_name=""):
    grid = np.array(step_data['grid'])

    visited_raw = step_data.get('visited')
    if isinstance(visited_raw, list) and visited_raw and isinstance(visited_raw[0], tuple):
        # visited is a list of (x, y)
        visited = np.zeros_like(grid)
        for x, y in visited_raw:
            visited[x][y] = 1
    elif isinstance(visited_raw, (np.ndarray, list)):
        # visited is a 2D array/matrix
        visited = np.array(visited_raw)
    else:
        visited = np.zeros_like(grid)

    path = step_data.get('path', [])
    frontier = step_data.get('frontier', [])
    fig, ax = plt.subplots(figsize=(5,5))
    cmap = plt.cm.get_cmap('Greys')
    ax.imshow(grid, cmap=cmap, alpha=0.7)
    visited_mask = (visited == 1)
    ax.scatter(np.where(visited_mask)[1], np.where(visited_mask)[0], c='cyan', s=100, marker='s', label='Visited', alpha=0.3)
    if path:
        px, py = zip(*path)
        ax.plot(py, px, color='yellow', linewidth=3, label='Path')
    for fx, fy in frontier:
        ax.scatter(fy, fx, c='red', s=100, marker='o', label='Frontier')
    ax.scatter(start[1], start[0], c='green', s=200, marker='o', label='Start')
    ax.scatter(end[1], end[0], c='blue', s=200, marker='*', label='End')
    ax.set_title(f"{algo_name} Pathfinding Step")
    ax.set_xticks([])
    ax.set_yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    st.pyplot(fig)

def visualize_bst_from_node(root):
    if root is None:
        st.write("Empty BST")
        return

    G = nx.DiGraph()

    def add_edges(node):
        if node is None:
            return
        G.add_node(node.val)
        if node.left:
            G.add_edge(node.val, node.left.val)
            add_edges(node.left)
        if node.right:
            G.add_edge(node.val, node.right.val)
            add_edges(node.right)

    add_edges(root)

    pos = hierarchy_pos(G, root.val)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=14, arrows=False)
    plt.title("BST Structure")
    st.pyplot(plt.gcf())
    plt.clf()

def hierarchy_pos(G, root, width=1.5, vert_gap=0.3, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None):
    """
    If there is a cycle that is reachable from root, then this will see infinite recursion.
    G: the graph (must be a tree)
    root: the root node of current branch
    width: horizontal space allocated for drawing
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                parent=root)
    return pos

    pos = hierarchy_pos(G, root.val, width=2.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, labels=labels, with_labels=True, arrows=False,
            node_color='deepskyblue', node_size=1300, ax=ax)
    ax.set_title("BST (structure)")
    ax.axis('off')
    st.pyplot(fig)

def visualize_kruskal_step(nodes, all_edges, mst_edges_so_far):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(all_edges)

    pos = nx.spring_layout(G, seed=42)

    # Draw all edges in light color
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='gray', alpha=0.4)
    # Draw MST edges so far in bold color
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges_so_far, edge_color='green', width=3)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='deepskyblue', node_size=900)
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_color='black')
    # Draw edge weights
    edge_labels = {(u, v): w for u, v, w in all_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Kruskal's MST - Animation Step")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def visualize_prim_step(nodes, all_edges, mst_edges_so_far, visited_nodes=None):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(all_edges)

    pos = nx.spring_layout(G, seed=42)

    # Draw all edges in light color
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='gray', alpha=0.4)
    # Draw MST edges so far in bold color
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges_so_far, edge_color='green', width=3)
    # Draw nodes, highlighting visited ones if provided
    node_colors = ['lightgreen' if visited_nodes and n in visited_nodes else 'deepskyblue' for n in nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=16, font_color='black')
    # Draw edge weights
    edge_labels = {(u, v): w for u, v, w in all_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Prim's MST - Animation Step")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()