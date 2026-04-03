"""
ENPM661 - Spring 2026
Project 03 - Phase 1: Backward A* Algorithm for Mobile Robot Path Planning

Team Members : Jigar Shah, Syed Ahmed, Sidharth Mathur
GitHub Repo  : https://github.com/sid-mat/backward-astar-algo

Description
-----------
Implements backward A* to find the optimal path for a differential-drive
mobile robot (radius 5 mm, clearance 5 mm) on a 600 x 250 mm map.

Action Set  : 5 moves — heading ± 0°, ± 30°, ± 60° then forward by step L.
Duplicate detection : visited matrix V[1200 x 500 x 12]  (per project spec).
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

# ═══════════════════════════════════════════════════════════════════════════════
#  MAP CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MAP_W  = 600   # mm (width)
MAP_H  = 250   # mm (height)

VIZ_SCALE = 3  # upscale factor for display/video

# ── Duplicate-node thresholds (project spec) ──────────────────────────────────
XY_THRESH    = 0.5   # spatial threshold (units)
TH_THRESH    = 30    # angular threshold (degrees)
N_THETA      = int(360 // TH_THRESH)  # 12 discrete angle bins

# ── Visited-matrix dimensions (project spec §Step 03) ─────────────────────────
# V[250/0.5 × 600/0.5 × 12]  =  V[500 × 1200 × 12]
# BUT the spec's index examples show:
#   node(x=3.2, y=4.7) → V[6][9][0]   → ix=round(x/0.5)=6,  iy=round(y/0.5)=9
#   node(x=10.2,y=8.8) → V[20][18][1] → ix=round(x/0.5)=20, iy=round(y/0.5)=18
# So the first index tracks x (range 0..1199), second tracks y (range 0..499).
VIX_MAX = int(MAP_W  / XY_THRESH)   # 1200  (x direction)
VIY_MAX = int(MAP_H  / XY_THRESH)   #  500  (y direction)
VT_MAX  = N_THETA                    #   12

# ── Goal/start proximity threshold ───────────────────────────────────────────
GOAL_THRESH = 1.5   # units

# ── BGR colours ───────────────────────────────────────────────────────────────
C_FREE    = (255, 255, 255)
C_OBS     = (0,   0,   180)
C_START   = (0,   200, 0  )
C_GOAL    = (0,   0,   220)
C_EXPLORE = (30,  200, 230)   # yellow-ish arrows during exploration
C_PATH    = (0,   0,   255)   # bright red optimal path


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
    """Return True if (x,y) is inside an obstacle or its clearance zone."""
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
    r1     = ((150 - c) <= X) & (X <= (175 + c)) & (Y <= (125 + c))
    r2     = ((250 - c) <= X) & (X <= (275 + c)) & (Y >= (125 - c))
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
    ycar = (MAP_H - 1 - np.arange(MAP_H, dtype=np.float32))
    X, Y = np.meshgrid(cols, ycar)
    c = clearance
    border = (X < c) | (X > MAP_W - c) | (Y < c) | (Y > MAP_H - c)
    r1     = ((150 - c) <= X) & (X <= (175 + c)) & (Y <= (125 + c))
    r2     = ((250 - c) <= X) & (X <= (275 + c)) & (Y >= (125 - c))
    circle = (X - 400) ** 2 + (Y - 110) ** 2 <= (50 + c) ** 2
    obs    = (border | r1 | r2 | circle)
    canvas = np.full((MAP_H, MAP_W, 3), C_FREE, dtype=np.uint8)
    canvas[obs] = C_OBS
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cart_to_img(x: float, y: float):
    """Cartesian (x,y) → OpenCV pixel (col, row)."""
    return int(round(x)), MAP_H - 1 - int(round(y))


# ═══════════════════════════════════════════════════════════════════════════════
#  VISITED-MATRIX INDEXING  (per project spec §Step 03)
# ═══════════════════════════════════════════════════════════════════════════════

def visited_idx(x: float, y: float, theta: float):
    """
    Map (x, y, theta) → (ix, iy, it) for the visited matrix.

    Per spec examples:
        node(x=3.2,  y=4.7,  θ=0°)  → V[6][9][0]
        node(x=10.2, y=8.8,  θ=30°) → V[20][18][1]
    So: ix = round(x/0.5),  iy = round(y/0.5),  it = round(θ/30) % 12.
    """
    ix = int(round(x / XY_THRESH))
    iy = int(round(y / XY_THRESH))
    it = int(round(theta / TH_THRESH)) % VT_MAX
    ix = max(0, min(VIX_MAX - 1, ix))
    iy = max(0, min(VIY_MAX - 1, iy))
    return ix, iy, it


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTION SET  — 5 functions, one per action (project spec §Step 01)
#
#  Action i: rotate by Δθ_i, then move forward by L.
#    x' = x + L·cos(θ + Δθ)
#    y' = y + L·sin(θ + Δθ)
#    θ' = (θ + Δθ) mod 360
#  Cost of every action = L  (Euclidean distance moved)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply(x, y, theta, delta, L):
    nt  = (theta + delta) % 360
    rad = math.radians(nt)
    return x + L * math.cos(rad), y + L * math.sin(rad), nt


def action_straight(x, y, theta, L):       return _apply(x, y, theta,   0, L)
def action_left30(x, y, theta, L):         return _apply(x, y, theta,  30, L)
def action_right30(x, y, theta, L):        return _apply(x, y, theta, -30, L)
def action_left60(x, y, theta, L):         return _apply(x, y, theta,  60, L)
def action_right60(x, y, theta, L):        return _apply(x, y, theta, -60, L)

ALL_ACTIONS = [
    action_straight,
    action_left30,
    action_right30,
    action_left60,
    action_right60,
]


def get_neighbours(x, y, theta, L, obs_grid):
    """Apply all 5 actions; return valid (nx, ny, ntheta, cost) tuples."""
    result = []
    for act in ALL_ACTIONS:
        nx, ny, nt = act(x, y, theta, L)
        if not obs_fast(nx, ny, obs_grid):
            result.append((nx, ny, nt, float(L)))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  HEURISTIC  (Euclidean distance — consistent, admissible)
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  GOAL-PROXIMITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def within_threshold(x, y, tx, ty, threshold=GOAL_THRESH):
    return euclidean(x, y, tx, ty) <= threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD A* SEARCH
#
#  The search is seeded with the GOAL node.
#  The heuristic at each node x is dist(x, START)  ← estimates cost-to-come
#    when searching backward (i.e., to reach the start from x).
#  We stop when we pop a node within GOAL_THRESH of the START.
#  Backtracking parent pointers reconstructs the forward path start → goal.
# ═══════════════════════════════════════════════════════════════════════════════

def backward_astar(start, goal, step_size, obs_grid):
    """
    Parameters
    ----------
    start      : (sx, sy, stheta)
    goal       : (gx, gy, gtheta)
    step_size  : integer 1–10
    obs_grid   : pre-computed obstacle look-up table

    Returns
    -------
    path           : list[(x,y,theta)] in start→goal order, or None
    explored_edges : list[((x1,y1),(x2,y2))] for animation
    """
    sx, sy, stheta = start
    gx, gy, gtheta = goal

    # ── Visited (closed-set) matrix: shape (VIX_MAX, VIY_MAX, VT_MAX) ─────────
    # ix from x (0..1199), iy from y (0..499)
    visited = np.zeros((VIX_MAX, VIY_MAX, VT_MAX), dtype=np.uint8)

    # ── Book-keeping dicts ─────────────────────────────────────────────────────
    # key = (ix, iy, it) tuple
    parent_map = {}   # key → (parent_key | None,  (x, y, theta))
    cost_g     = {}   # key → best g-cost so far

    counter = 0   # tie-breaker in heap

    # ── Seed with GOAL node ────────────────────────────────────────────────────
    gk = visited_idx(gx, gy, gtheta)
    cost_g[gk]     = 0.0
    parent_map[gk] = (None, (gx, gy, gtheta))
    h0             = euclidean(gx, gy, sx, sy)
    heap           = [(h0, counter, gx, gy, gtheta)]
    counter       += 1

    explored_edges = []
    found          = False
    final_node     = None

    # ── Main search loop ───────────────────────────────────────────────────────
    while heap:
        f_cur, _, cx, cy, ctheta = heapq.heappop(heap)
        ck = visited_idx(cx, cy, ctheta)
        cix, ciy, cit = ck

        # Skip stale heap entries (already closed)
        if visited[cix, ciy, cit]:
            continue

        # Close this node
        visited[cix, ciy, cit] = 1

        # ── Check if we have reached the START vicinity ────────────────────────
        if within_threshold(cx, cy, sx, sy):
            found      = True
            final_node = (cx, cy, ctheta)
            break

        # ── Expand neighbours ──────────────────────────────────────────────────
        for nx, ny, ntheta, move_cost in get_neighbours(cx, cy, ctheta, step_size, obs_grid):
            nk = visited_idx(nx, ny, ntheta)
            nix, niy, nit = nk

            if visited[nix, niy, nit]:
                continue

            new_g = cost_g[ck] + move_cost
            if new_g < cost_g.get(nk, float('inf')):
                cost_g[nk]     = new_g
                parent_map[nk] = (ck, (nx, ny, ntheta))
                h              = euclidean(nx, ny, sx, sy)
                heapq.heappush(heap, (new_g + h, counter, nx, ny, ntheta))
                counter       += 1
                explored_edges.append(((cx, cy), (nx, ny)))

    if not found:
        return None, explored_edges

    # ── Backtrack: final_node(≈start) → … → goal ─────────────────────────────
    # parent_map was seeded at goal, expanded outward.
    # So the chain:  final_node.parent → … → goal_node (parent=None)
    # Traversal produces: [final_node, …, goal]  which is start→goal order.
    path = []
    cur  = visited_idx(*final_node)
    while cur is not None:
      if cur not in parent_map:   # Correction: stops instead of crashing
          break
      parent_key, xyz = parent_map[cur]
      path.append(xyz)
      cur = parent_key
    # path = [~start, intermediate…, goal]  ✓

    return path, explored_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION  — node exploration + optimal path, saved to .mp4
#
#  Per project spec §Step 05:
#    "The visualization … should start ONLY AFTER the exploration is complete
#     and optimal path is found."
# ═══════════════════════════════════════════════════════════════════════════════

def draw_arrow(canvas, x1, y1, x2, y2, color, thickness=1):
    """Draw an arrow from Cartesian (x1,y1)→(x2,y2) on the canvas."""
    p1 = cart_to_img(x1, y1)
    p2 = cart_to_img(x2, y2)
    if p1 != p2:
        cv2.arrowedLine(canvas, p1, p2, color, thickness, tipLength=0.4)


def visualise(canvas_base, start, goal, explored_edges, path,
              video_path="output_path.mp4"):
    """
    Render the full animation and save to video.
    Phase 1 — draw all explored edges (batched for manageable video length).
    Phase 2 — draw the optimal path over the exploration.
    """
    h, w   = canvas_base.shape[:2]
    out_h  = h * VIZ_SCALE
    out_w  = w * VIZ_SCALE

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = 60
    writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

    canvas = canvas_base.copy()

    # ── Mark start and goal ────────────────────────────────────────────────────
    sx, sy, _ = start
    gx, gy, _ = goal
    cv2.circle(canvas, cart_to_img(sx, sy), 6, C_START, -1)
    cv2.circle(canvas, cart_to_img(gx, gy), 6, C_GOAL,  -1)
    cv2.circle(canvas, cart_to_img(gx, gy), int(GOAL_THRESH + 0.5), C_GOAL, 1)

    n_edges = len(explored_edges)
    print(f"\n  Animating {n_edges:,} exploration edges …")

    # ── Phase 1: exploration ───────────────────────────────────────────────────
    # Aim for ~20-second exploration clip at 60 fps → 1200 frames total.
    BATCH = max(1, n_edges // (fps * 20))
    for i, ((x1, y1), (x2, y2)) in enumerate(explored_edges):
        draw_arrow(canvas, x1, y1, x2, y2, C_EXPLORE, thickness=1)
        if i % BATCH == 0:
            frame = cv2.resize(canvas, (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST)
            writer.write(frame)

    # Hold exploration frame 1 s
    frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    for _ in range(fps):
        writer.write(frame)

    # ── Phase 2: optimal path ─────────────────────────────────────────────────
    print(f"  Animating {len(path)} path waypoints …")
    for i in range(1, len(path)):
        x1, y1, _ = path[i - 1]
        x2, y2, _ = path[i]
        draw_arrow(canvas, x1, y1, x2, y2, C_PATH, thickness=3)
        frame = cv2.resize(canvas, (out_w, out_h),
                           interpolation=cv2.INTER_NEAREST)
        writer.write(frame)

    # Hold final frame 2 s
    frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    for _ in range(fps * 2):
        writer.write(frame)

    writer.release()
    print(f"  Video saved → {video_path}")

    cv2.imshow("Backward A* — Result  (press any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
#  USER INPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
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
            print("  ✗  θ must be a multiple of 30  (e.g. -60, -30, 0, 30, 60, …).")
        except ValueError:
            print("  ✗  Please enter an integer.")


def ask_point(label, clearance):
    """Keep prompting until a free-space (x, y, θ) is entered."""
    while True:
        print(f"\n  ── {label} ──────────────────────────────────────────")
        x     = ask_float(f"    x   (0 – {MAP_W}):   ")
        y     = ask_float(f"    y   (0 – {MAP_H}): ")
        theta = ask_theta( "    θ   (multiple of 30°): ")

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

    # ── [1] Robot parameters ──────────────────────────────────────────────────
    print("\n[1/5]  Robot Parameters")
    robot_r  = ask_float("  Robot radius   (mm) [default 5]: ") or 5.0
    user_c   = ask_float("  User clearance (mm) [default 5]: ") or 5.0
    total_c  = robot_r + user_c
    print(f"  Total clearance = {robot_r} + {user_c} = {total_c} mm")

    step_L   = ask_int_range("\n  Step size L  (1 – 10 units): ", 1, 10)

    # ── [2] Start & goal ──────────────────────────────────────────────────────
    print("\n[2/5]  Start & Goal  (Cartesian, origin = bottom-left corner)")
    start = ask_point("Start Point", total_c)
    goal  = ask_point("Goal Point",  total_c)

    # ── [3] Build map ─────────────────────────────────────────────────────────
    print("\n[3/5]  Building obstacle map …")
    t0       = time.time()
    obs_grid = build_obstacle_grid(total_c)
    canvas   = build_map_image(total_c)
    print(f"       Done in {time.time() - t0:.3f}s  (canvas shape: {canvas.shape})")

    # ── [4] Backward A* ───────────────────────────────────────────────────────
    print("\n[4/5]  Running Backward A* search …")
    t1 = time.time()
    path, explored = backward_astar(start, goal, step_L, obs_grid)
    dt = time.time() - t1
    print(f"       Search time    : {dt:.3f}s")
    print(f"       Edges explored : {len(explored):,}")

    if path is None:
        print("\n  ✗  No path found between the given start and goal.")
        print("     Suggestions: try positions farther from obstacles,")
        print("     a larger step size, or a smaller clearance value.")
        return

    cost = sum(
        euclidean(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        for i in range(len(path) - 1)
    )
    print(f"       Path waypoints : {len(path)}")
    print(f"       Path cost      : {cost:.3f} units")

    # ── [5] Visualise ─────────────────────────────────────────────────────────
    print("\n[5/5]  Generating animation …")
    visualise(canvas, start, goal, explored, path, video_path="output_path.mp4")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Done!  Total runtime: {time.time() - t0:.2f}s".ljust(62) + "  ║")
    print("║  Output: output_path.mp4                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
