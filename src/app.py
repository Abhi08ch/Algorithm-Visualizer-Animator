import streamlit as st
import numpy as np
import time
from src_algorithms import (
    bubble_sort_steps, binary_search_steps, bfs_pathfinding_steps,
    dfs_pathfinding_steps, dijkstra_pathfinding_steps, prim_steps,
    kruskal_steps, bst_insert_steps
)
from src_visualizer import (
    visualize_array, visualize_binary_search, visualize_grid_step, visualize_bst_from_node, visualize_kruskal_step, visualize_prim_step
)

st.set_page_config(layout="wide")
st.title("🧮 Algorithm Visualizer & Animator")

algo_choice = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Bubble Sort",
        "Binary Search",
        "BFS Pathfinding",
        "DFS Pathfinding",
        "Dijkstra Pathfinding",
        "Prim MST",
        "Kruskal MST",
        "BST Insert"
    ]
)

if "step" not in st.session_state:
    st.session_state.step = 0
if "running" not in st.session_state:
    st.session_state.running = False

def start_animation(max_steps):
    st.session_state.running = True

def stop_animation():
    st.session_state.running = False

def reset_animation():
    st.session_state.running = False
    st.session_state.step = 0

def sidebar_info(desc):
    st.sidebar.markdown(f"**ℹ️ {desc}**")

# --- Algorithm blocks ---

if algo_choice == "Bubble Sort":
    sidebar_info("Animated Bubble Sort with step-by-step visualization.")
    arr_len = st.sidebar.slider("Array Length", 5, 12, 8)
    arr = np.random.permutation(arr_len).tolist()
    steps = list(bubble_sort_steps(arr))
    max_steps = len(steps) - 1

    if st.session_state.step > max_steps:
        st.session_state.step = 0
        st.session_state.running = False

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Bubble Sort Animation")
        visualize_array(steps[st.session_state.step], st.session_state.step)
    with col2:
        st.markdown("**Controls**")
        st.button("▶️ Run", on_click=lambda: start_animation(max_steps), disabled=st.session_state.running)
        st.button("⏸ Pause", on_click=stop_animation)
        st.button("⏮ Reset", on_click=reset_animation)
        if not st.session_state.running:
            st.session_state.step = st.slider(
                "Step", 0, max_steps, st.session_state.step, key="step_slider_bubble"
            )
        st.info(f"Current Array: {steps[st.session_state.step]}")

    if st.session_state.running and st.session_state.step < max_steps:
        time.sleep(0.5)
        st.session_state.step += 1
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    elif st.session_state.running and st.session_state.step >= max_steps:
        st.session_state.running = False

elif algo_choice == "Binary Search":
    sidebar_info("Visualize Binary Search searching for a target value.")
    arr = sorted(np.random.randint(1, 100, 10).tolist())
    target = st.sidebar.number_input("Target Value", min_value=min(arr), max_value=max(arr), value=arr[len(arr)//2])
    steps = binary_search_steps(arr, target)
    max_steps = len(steps) - 1

    if st.session_state.step > max_steps:
        st.session_state.step = 0
        st.session_state.running = False

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Binary Search Animation")
        s = steps[st.session_state.step]
        visualize_binary_search(s["array"], s["left"], s["right"], s["mid"])
    with col2:
        st.markdown("**Controls**")
        st.button("▶️ Run", on_click=lambda: start_animation(max_steps), disabled=st.session_state.running)
        st.button("⏸ Pause", on_click=stop_animation)
        st.button("⏮ Reset", on_click=reset_animation)
        if not st.session_state.running:
            st.session_state.step = st.slider(
                "Step", 0, max_steps, st.session_state.step, key="step_slider_bs"
            )
        st.info(f"Array: {arr}")

    if st.session_state.running and st.session_state.step < max_steps:
        time.sleep(1)
        st.session_state.step += 1
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    elif st.session_state.running and st.session_state.step >= max_steps:
        st.session_state.running = False

elif algo_choice in ["BFS Pathfinding", "DFS Pathfinding", "Dijkstra Pathfinding"]:
    sidebar_info(f"Animated {algo_choice} on a random grid.")
    grid_size = st.sidebar.slider("Grid Size", 5, 15, 8)
    np.random.seed(st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1))
    p_obstacle = st.sidebar.slider("Obstacle Probability", 0.0, 0.6, 0.2)
    grid = (np.random.rand(grid_size, grid_size) < p_obstacle).astype(int)
    start = (0, 0)
    end = (grid_size-1, grid_size-1)
    grid[start[0]][start[1]] = 0
    grid[end[0]][end[1]] = 0
    if algo_choice == "BFS Pathfinding":
        steps = bfs_pathfinding_steps(grid.tolist(), start, end)
    elif algo_choice == "DFS Pathfinding":
        steps = dfs_pathfinding_steps(grid.tolist(), start, end)
    else:
        steps = dijkstra_pathfinding_steps(grid.tolist(), start, end)
    max_steps = len(steps) - 1

    if st.session_state.step > max_steps:
        st.session_state.step = 0
        st.session_state.running = False

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader(f"{algo_choice} Animation")
        visualize_grid_step(steps[st.session_state.step], start, end, algo_choice)
    with col2:
        st.markdown("**Controls**")
        st.button("▶️ Run", on_click=lambda: start_animation(max_steps), disabled=st.session_state.running)
        st.button("⏸ Pause", on_click=stop_animation)
        st.button("⏮ Reset", on_click=reset_animation)
        if not st.session_state.running:
            st.session_state.step = st.slider(
                "Step", 0, max_steps, st.session_state.step, key="step_slider_grid"
            )
        st.dataframe(grid)
        st.info("0=open, 1=obstacle")

    if st.session_state.running and st.session_state.step < max_steps:
        time.sleep(0.7)
        st.session_state.step += 1
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    elif st.session_state.running and st.session_state.step >= max_steps:
        st.session_state.running = False

elif algo_choice == "Prim MST":
    sidebar_info("Visualize Prim's MST algorithm step-by-step.")
    n = st.sidebar.slider("Nodes", 3, 8, 5)
    np.random.seed(st.sidebar.number_input("Random Seed", value=42, step=1))
    graph = {i: [] for i in range(n)}
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = np.random.randint(1, 20)
            graph[i].append((j, w))
            graph[j].append((i, w))
            all_edges.append((i, j, w))
    steps = prim_steps(graph, 0)
    max_steps = len(steps)-1

    # Clamp the step to valid range
    if st.session_state.step > max_steps:
        st.session_state.step = 0
        st.session_state.running = False

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Prim's MST Animation")
        mst_edges = steps[st.session_state.step]["mst"]  # Should be [(u,v,w)]
        visited = steps[st.session_state.step].get("visited", [])
        visualize_prim_step(list(range(n)), all_edges, mst_edges, visited)
        st.write("MST Edges:", mst_edges)
        st.write("Visited Nodes:", visited)
    with col2:
        st.markdown("**Controls**")
        st.button("▶️ Run", on_click=lambda: start_animation(max_steps), disabled=st.session_state.running)
        st.button("⏸ Pause", on_click=stop_animation)
        st.button("⏮ Reset", on_click=reset_animation)
        if not st.session_state.running:
            st.session_state.step = st.slider(
                "Step", 0, max_steps, st.session_state.step, key="step_slider_prim"
            )
        st.write("Graph", graph)

    if st.session_state.running and st.session_state.step < max_steps:
        time.sleep(0.7)
        st.session_state.step += 1
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    elif st.session_state.running and st.session_state.step >= max_steps:
        st.session_state.running = False

elif algo_choice == "Kruskal MST":
    sidebar_info("Visualize Kruskal's MST algorithm step-by-step.")
    n = st.sidebar.slider("Nodes", 3, 8, 5)
    np.random.seed(st.sidebar.number_input("Random Seed", value=42, step=1))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = np.random.randint(1, 20)
            edges.append((i, j, w))
    steps = kruskal_steps(edges, n)
    max_steps = len(steps)-1

    if st.session_state.step > max_steps:
        st.session_state.step = 0
        st.session_state.running = False

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Kruskal's MST Animation")
        mst_edges = steps[st.session_state.step]["mst"]  # Should be a list of (u,v,w)
        visualize_kruskal_step(list(range(n)), edges, mst_edges)
        st.write("MST Edges so far:", mst_edges)
        st.write("Union-Find Parent:", steps[st.session_state.step]["parent"])
    with col2:
        st.markdown("**Controls**")
        st.button("▶️ Run", on_click=lambda: start_animation(max_steps), disabled=st.session_state.running)
        st.button("⏸ Pause", on_click=stop_animation)
        st.button("⏮ Reset", on_click=reset_animation)
        if not st.session_state.running:
            st.session_state.step = st.slider(
                "Step", 0, max_steps, st.session_state.step, key="step_slider_kruskal"
            )
        st.write("Edges", edges)

    if st.session_state.running and st.session_state.step < max_steps:
        time.sleep(0.7)
        st.session_state.step += 1
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    elif st.session_state.running and st.session_state.step >= max_steps:
        st.session_state.running = False

elif algo_choice == "BST Insert":
    sidebar_info("Visualize BST insertions and the evolving tree.")
    n = st.sidebar.slider("Node Count", 3, 12, 7, key="bst_node_count")
    seed = st.sidebar.number_input("Random Seed", value=42, step=1, key="bst_seed")
    np.random.seed(seed)

# --- Session State Initialization ---
if "bst_values" not in st.session_state:
    st.session_state.bst_values = None
if "bst_steps" not in st.session_state:
    st.session_state.bst_steps = None
if "bst_step" not in st.session_state:
    st.session_state.bst_step = 0
if "bst_running" not in st.session_state:
    st.session_state.bst_running = False

def reset_bst():
    st.session_state.bst_values = np.random.choice(range(1, 50), n, replace=False).tolist()
    st.session_state.bst_steps = bst_insert_steps(st.session_state.bst_values)
    st.session_state.bst_step = 0
    st.session_state.bst_running = False

def run_bst():
    st.session_state.bst_running = True

def pause_bst():
    st.session_state.bst_running = False

# --- Automatically reset when node count or seed changes ---
if st.session_state.bst_values is None or st.session_state.bst_steps is None \
        or len(st.session_state.bst_values) != n:
    reset_bst()

# --- UI ---
col1, col2 = st.columns([2, 1])
max_steps = len(st.session_state.bst_steps) - 1

# Clamp step if out of range
if st.session_state.bst_step > max_steps:
    st.session_state.bst_step = 0
    st.session_state.bst_running = False

with col1:
    st.subheader("BST Insert Animation")
    step = st.session_state.bst_steps[st.session_state.bst_step]
    visualize_bst_from_node(step["root"])
with col2:
    st.markdown("**Controls**")
    st.button("▶️ Run", on_click=run_bst, disabled=st.session_state.bst_running, key="bst_run_btn")
    st.button("⏸ Pause", on_click=pause_bst, key="bst_pause_btn")
    st.button("⏮ Reset", on_click=reset_bst, key="bst_reset_btn")
    if not st.session_state.bst_running:
        st.session_state.bst_step = st.slider(
            "Step", 0, max_steps, st.session_state.bst_step, key="bst_step_slider"
        )
    st.info(f"Inserted values: {st.session_state.bst_values}")

# --- Animation logic ---
if st.session_state.bst_running and st.session_state.bst_step < max_steps:
    time.sleep(1)
    st.session_state.bst_step += 1
    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
elif st.session_state.bst_running and st.session_state.bst_step >= max_steps:
    st.session_state.bst_running = False