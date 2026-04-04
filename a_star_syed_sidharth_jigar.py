""""
ENPM661 - Spring 2026
Project 03 - Phase 1: Backward A* Algorithm for Mobile Robot Path Planning
 
Team Members : Syed Ahmed, Sidharth Mathur, Jigar Shah
GitHub Repo  : https://github.com/sid-mat/backward-astar-algo
 
Description
-----------
Implements backward A* to find the optimal path for a differential-drive
mobile robot (radius 5 mm, clearance 5 mm) on a 600 x 250 mm map.
 
Action Set  : 5 moves — heading ± 0°, ± 30°, ± 60° then forward by step L.
Duplicate detection : visited matrix V[1200 x 500 x 12] (per project spec).
Heuristic   : Euclidean distance to the start node (backward search).
Goal check  : within 1.5 unit radius of the target point.
 
Deliverables generated
----------------------
  output_path.mp4  — animation (exploration + optimal path)
 
Dependencies
------------
  numpy, opencv-python
  pip install numpy opencv-python
"""

import heapq
import math
import time
import numpy as np
import cv2

# Import the 5 action functions from the separate actions module ───
from actions import (
    action_straight, action_left30, action_right30,
    action_left60, action_right60,
    ALL_ACTIONS, euclidean, segment_is_free, get_neighbours
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAP CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MAP_W = 600   # mm (width)
MAP_H = 250   # mm (height)

VIZ_SCALE = 3  # upscale factor for display/video

# ── Duplicate-node thresholds (project spec) ──────────────────────────────────
XY_THRESH = 0.5    # spatial threshold (units)
TH_THRESH = 30     # angular threshold (degrees)
N_THETA = int(360 // TH_THRESH)  # 12 discrete angle bins

# ── Visited-matrix dimensions (project spec §Step 03) ─────────────────────────
# Spec matrix size = (500 x 1200 x 12), but the spec example indexes as V[x][y][t].
# This implementation stores visited as visited[ix, iy, it] for consistency with
# the provided examples:
#   node(x=3.2, y=4.7, θ=0°)  -> V[6][9][0]
#   node(x=10.2,y=8.8, θ=30°) -> V[20][18][1]
VIX_MAX = int(MAP_W / XY_THRESH)   # 1200 (x direction)
VIY_MAX = int(MAP_H / XY_THRESH)   #  500 (y direction)
VT_MAX = N_THETA                   #   12

# ── Goal/start proximity threshold ─────────────────────────────────────────────
GOAL_THRESH = 1.5   # units

# ── BGR colours ────────────────────────────────────────────────────────────────
C_FREE = (255, 255, 255)
C_OBS = (0, 0, 180)
C_START = (0, 200, 0)
C_GOAL = (0, 0, 220)
C_EXPLORE = (30, 200, 230)   # yellow-ish arrows during exploration
C_PATH = (0, 0, 255)         # bright red optimal path


# ═══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE / FREE-SPACE  (half-plane, semi-algebraic model)
#
#  Map layout — Project 2 (600 × 250 mm), Cartesian origin at bottom-left:
#    Rect-1 : x ∈ [150,175], y ∈ [0,125]     (bottom pillar)
#    Rect-2 : x ∈ [250,275], y ∈ [125,250]   (top pillar)
#    Circle : centre (400,110), radius 50
#    Border walls on all four sides
#
#  Every obstacle is inflated by c = robot_radius + user_clearance.
# ═══════════════════════════════════════════════════════════════════════════════

def is_obstacle(x: float, y: float, c: float) -> bool:
    """Return True if (x, y) is inside an obstacle or its clearance zone."""
    if x < c or x > MAP_W - c or y < c or y > MAP_H - c:
        return True
    if (150 - c) <= x <= (175 + c) and y <= (125 + c):
        return True
    if (250 - c) <= x <= (275 + c) and y >= (125 - c):
        return True
    if (x - 400) ** 2 + (y - 110) ** 2 <= (50 + c) ** 2:
        return True
    return False


def build_obstacle_grid(clearance: float) -> np.ndarray:
    """
    Pre-compute a boolean obstacle look-up table at XY_THRESH resolution.
    Shape: (VIY_MAX, VIX_MAX) = (500, 1200).
    grid[iy, ix] == 1  ↔  the 0.5-unit cell (ix*0.5, iy*0.5) is blocked.
    """
    xs = np.arange(VIX_MAX) * XY_THRESH   # x coords: 0, 0.5, …, 599.5
    ys = np.arange(VIY_MAX) * XY_THRESH   # y coords: 0, 0.5, …, 249.5
    X, Y = np.meshgrid(xs, ys)            # shape (500, 1200)
    c = clearance
    border = (X < c) | (X > MAP_W - c) | (Y < c) | (Y > MAP_H - c)
    r1 = ((150 - c) <= X) & (X <= (175 + c)) & (Y <= (125 + c))
    r2 = ((250 - c) <= X) & (X <= (275 + c)) & (Y >= (125 - c))
    circle = (X - 400) ** 2 + (Y - 110) ** 2 <= (50 + c) ** 2
    return (border | r1 | r2 | circle).astype(np.uint8)


def obs_fast(x: float, y: float, grid: np.ndarray) -> bool:
    """O(1) obstacle check using pre-computed grid."""
    iy = int(round(y / XY_THRESH))
    ix = int(round(x / XY_THRESH))
    if ix < 0 or ix >= VIX_MAX or iy < 0 or iy >= VIY_MAX:
        return True
    return bool(grid[iy, ix])


def build_map_image(clearance: float) -> np.ndarray:
    """
    Build an RGB OpenCV canvas (H=250, W=600) with obstacles coloured.
    Uses vectorised NumPy; row-0 is the top of the image.
    """
    cols = np.arange(MAP_W, dtype=np.float32)
    ycar = MAP_H - 1 - np.arange(MAP_H, dtype=np.float32)
    X, Y = np.meshgrid(cols, ycar)
    c = clearance
    border = (X < c) | (X > MAP_W - c) | (Y < c) | (Y > MAP_H - c)
    r1 = ((150 - c) <= X) & (X <= (175 + c)) & (Y <= (125 + c))
    r2 = ((250 - c) <= X) & (X <= (275 + c)) & (Y >= (125 - c))
    circle = (X - 400) ** 2 + (Y - 110) ** 2 <= (50 + c) ** 2
    obs = border | r1 | r2 | circle
    canvas = np.full((MAP_H, MAP_W, 3), C_FREE, dtype=np.uint8)
    canvas[obs] = C_OBS
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cart_to_img(x: float, y: float):
    """Cartesian (x, y) → OpenCV pixel (col, row)."""
    return int(round(x)), MAP_H - 1 - int(round(y))


# ═══════════════════════════════════════════════════════════════════════════════
#  VISITED-MATRIX INDEXING  (per project spec §Step 03)
# ═══════════════════════════════════════════════════════════════════════════════

def visited_idx(x: float, y: float, theta: float):
    """
    Map (x, y, theta) → (ix, iy, it) for the visited matrix.
    """
    ix = int(round(x / XY_THRESH))
    iy = int(round(y / XY_THRESH))
    it = int(round(theta / TH_THRESH)) % VT_MAX
    ix = max(0, min(VIX_MAX - 1, ix))
    iy = max(0, min(VIY_MAX - 1, iy))
    return ix, iy, it



# ═══════════════════════════════════════════════════════════════════════════════
#  ACTION SET  — [CHANGED] moved to actions.py; imported at the top of this file
#
#  Action i: rotate by Δθ_i, then move forward by L.
#    x' = x + L·cos(θ + Δθ)
#    y' = y + L·sin(θ + Δθ)
#    θ' = (θ + Δθ) mod 360
#  Cost of every action = L
#
#  The 5 functions (action_straight, action_left30, action_right30,
#  action_left60, action_right60), ALL_ACTIONS list, euclidean(),
#  segment_is_free(), and get_neighbours() are all defined in actions.py.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  GOAL-PROXIMITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def within_threshold(x, y, tx, ty, threshold=GOAL_THRESH):
    return euclidean(x, y, tx, ty) <= threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD A* SEARCH
#
#  The search is seeded with the GOAL node.
#  The heuristic at each node x is dist(x, START).
#  We stop when we pop a node within GOAL_THRESH of the START.
#  Backtracking parent pointers reconstructs the forward path start → goal.
# ═══════════════════════════════════════════════════════════════════════════════

def backward_astar(start, goal, step_size, obs_grid):
    sx, sy, stheta = start
    gx, gy, gtheta = goal

    visited = np.zeros((VIX_MAX, VIY_MAX, VT_MAX), dtype=np.uint8)
    parent_map = {}
    cost_g = {}
    counter = 0

    gk = visited_idx(gx, gy, gtheta)
    cost_g[gk] = 0.0
    parent_map[gk] = (None, (gx, gy, gtheta))
    h0 = euclidean(gx, gy, sx, sy)
    heap = [(h0, counter, gx, gy, gtheta)]
    counter += 1

    # Trivial case: goal is already close to start
    if within_threshold(gx, gy, sx, sy):
        print("  Start and goal are within threshold — trivial path.")
        return [(sx, sy, stheta), (gx, gy, gtheta)], []

    explored_edges = []
    final_node = None

    while heap:
        _, _, cx, cy, ctheta = heapq.heappop(heap)
        ck = visited_idx(cx, cy, ctheta)
        cix, ciy, cit = ck

        if visited[cix, ciy, cit]:
            continue

        visited[cix, ciy, cit] = 1

        if within_threshold(cx, cy, sx, sy):
            final_node = (cx, cy, ctheta)
            break

        for nx, ny, ntheta, move_cost in get_neighbours(cx, cy, ctheta, step_size, obs_grid, obs_fast):
            nk = visited_idx(nx, ny, ntheta)
            nix, niy, nit = nk

            if visited[nix, niy, nit]:
                continue

            new_g = cost_g[ck] + move_cost
            if new_g < cost_g.get(nk, float("inf")):
                cost_g[nk] = new_g
                parent_map[nk] = (ck, (nx, ny, ntheta))
                explored_edges.append(((cx, cy), (nx, ny)))
 
                # ── Goal check at generation time (per project spec) ──────────
                if within_threshold(nx, ny, sx, sy):
                    final_node = (nx, ny, ntheta)
                    break
                
                h = euclidean(nx, ny, sx, sy)
                heapq.heappush(heap, (new_g + h, counter, nx, ny, ntheta))
                counter += 1
                explored_edges.append(((cx, cy), (nx, ny)))

        if final_node is not None:
           break
 
    if final_node is None:
        return None, explored_edges

    # Backtrack from the popped near-start node to the seeded goal node.
    path = []
    cur = visited_idx(*final_node)
    while cur is not None:
        parent_key, xyz = parent_map[cur]
        path.append(xyz)
        cur = parent_key

    # Ensure the displayed path starts exactly at the user-entered start pose.
    first_x, first_y, first_theta = path[0]
    if (
        abs(first_x - sx) > 1e-6
        or abs(first_y - sy) > 1e-6
        or abs(first_theta - stheta) > 1e-6
    ):
        path.insert(0, (sx, sy, stheta))

    return path, explored_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION  — node exploration + optimal path, saved to .mp4
# ═══════════════════════════════════════════════════════════════════════════════

def draw_arrow(canvas, x1, y1, x2, y2, color, thickness=1):
    """Draw an arrow from Cartesian (x1,y1)→(x2,y2) on the canvas."""
    p1 = cart_to_img(x1, y1)
    p2 = cart_to_img(x2, y2)
    if p1 != p2:
        cv2.arrowedLine(canvas, p1, p2, color, thickness, tipLength=0.4)


def visualise(canvas_base, start, goal, explored_edges, path,
              video_path="output_path.mp4", show_window=False):
    """
    Render the full animation and save to video.
    Phase 1 — draw all explored edges.
    Phase 2 — draw the optimal path over the exploration.
    """
    h, w = canvas_base.shape[:2]
    out_h = h * VIZ_SCALE
    out_w = w * VIZ_SCALE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 60
    writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    canvas = canvas_base.copy()

    # Mark start and goal
    sx, sy, _ = start
    gx, gy, _ = goal
    cv2.circle(canvas, cart_to_img(sx, sy), 6, C_START, -1)
    cv2.circle(canvas, cart_to_img(gx, gy), 6, C_GOAL, -1)
    cv2.circle(canvas, cart_to_img(gx, gy), int(GOAL_THRESH + 0.5), C_GOAL, 1)

    n_edges = len(explored_edges)
    print(f"\n  Animating {n_edges:,} exploration edges …")

    # Phase 1: exploration
    batch = max(1, n_edges // (fps * 20)) if n_edges else 1
    for i, ((x1, y1), (x2, y2)) in enumerate(explored_edges):
        draw_arrow(canvas, x1, y1, x2, y2, C_EXPLORE, thickness=1)
        if i % batch == 0:
            frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            writer.write(frame)

    frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    for _ in range(fps):
        writer.write(frame)

    # Phase 2: optimal path
    print(f"  Animating {len(path)} path waypoints …")
    for i in range(1, len(path)):
        x1, y1, _ = path[i - 1]
        x2, y2, _ = path[i]
        draw_arrow(canvas, x1, y1, x2, y2, C_PATH, thickness=3)
        frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        writer.write(frame)

    frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    for _ in range(fps * 2):
        writer.write(frame)

    writer.release()
    print(f"  Video saved → {video_path}")

    if show_window:
        try:
            cv2.imshow("Backward A* — Result  (press any key to close)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("  OpenCV GUI display not available in this environment; video was still saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  USER INPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ask_float(prompt, default=None):
    """
    Ask for a float. If default is provided, pressing Enter returns that value.
    """
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  ✗  Please enter a number.")


def ask_int_range(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  ✗  Must be an integer between {lo} and {hi}.")
        except ValueError:
            print("  ✗  Please enter an integer.")


def ask_theta(prompt):
    """Accept any integer multiple of 30; normalise to [0, 360)."""
    while True:
        try:
            v = int(input(prompt))
            if v % 30 == 0:
                return float(v % 360)
            print("  ✗  θ must be a multiple of 30 (e.g. -60, -30, 0, 30, 60, …).")
        except ValueError:
            print("  ✗  Please enter an integer.")


def ask_point(label, clearance):
    """Keep prompting until a free-space (x, y, θ) is entered."""
    while True:
        print(f"\n  ── {label} ──────────────────────────────────────────")
        x = ask_float(f"    x   (0 – {MAP_W}): ")
        y = ask_float(f"    y   (0 – {MAP_H}): ")
        theta = ask_theta("    θ   (multiple of 30°): ")

        if not (0 <= x <= MAP_W and 0 <= y <= MAP_H):
            print("  ✗  Outside map bounds. Try again.")
            continue
        if is_obstacle(x, y, clearance):
            print(f"  ✗  ({x}, {y}) is inside an obstacle or clearance zone. Try again.")
            continue
        print(f"  ✓  {label} accepted: ({x}, {y}, {theta}°)")
        return float(x), float(y), theta


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ENPM661 Project 3 Phase 1 — Backward A* Path Planner       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # [1] Robot parameters
    print("\n[1/5]  Robot Parameters")
    robot_r = ask_float("  Robot radius   (mm) [default 5]: ", default=5.0)
    user_c = ask_float("  User clearance (mm) [default 5]: ", default=5.0)
    total_c = robot_r + user_c
    print(f"  Total clearance = {robot_r} + {user_c} = {total_c} mm")

    step_L = ask_int_range("\n  Step size L  (1 – 10 units): ", 1, 10)

    # [2] Start & goal
    print("\n[2/5]  Start & Goal  (Cartesian, origin = bottom-left corner)")
    start = ask_point("Start Point", total_c)
    goal = ask_point("Goal Point", total_c)

    # [3] Build map
    print("\n[3/5]  Building obstacle map …")
    build_start = time.time()
    obs_grid = build_obstacle_grid(total_c)
    canvas = build_map_image(total_c)
    print(f"       Done in {time.time() - build_start:.3f}s  (canvas shape: {canvas.shape})")

    # [4] Backward A*
    print("\n[4/5]  Running Backward A* search …")
    search_start = time.time()
    path, explored = backward_astar(start, goal, step_L, obs_grid)
    search_dt = time.time() - search_start
    print(f"       Search time    : {search_dt:.3f}s")
    print(f"       Edges explored : {len(explored):,}")

    if path is None:
        print("\n  ✗  No path found between the given start and goal.")
        print("     Suggestions: try positions farther from obstacles,")
        print("     a larger step size, or a smaller clearance value.")
        return

    cost = sum(
        euclidean(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        for i in range(len(path) - 1)
    )
    print(f"       Path waypoints : {len(path)}")
    print(f"       Path cost      : {cost:.3f} units")

    # [5] Visualise
    print("\n[5/5]  Generating animation …")
    visualise(
        canvas,
        start,
        goal,
        explored,
        path,
        video_path="output_path.mp4",
        show_window=False,
    )

    total_runtime = time.time() - build_start
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Done!  Total runtime: {total_runtime:.2f}s".ljust(62) + "  ║")
    print("║  Output: output_path.mp4                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
