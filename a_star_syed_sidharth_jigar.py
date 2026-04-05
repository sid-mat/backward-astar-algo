"""
ENPM661 Spring 2026 - Project 3 Phase 1
Backward A* path planner for a differential-drive mobile robot.

Team Members:
    Syed Yashal Ahmed  (sahmed43, 122288393)
    Sidharth Mathur    (sidmat03, 122277687)
    Jigar Shah         (jshah310, 121355690)

GitHub: https://github.com/sid-mat/backward-astar-algo

Map: 600 x 250 mm. Obstacle space spells "SM7687" using rectangular
strokes, each stroke defined as the intersection of 4 half-planes
(semi-algebraic model). Clearance inflates every stroke boundary by c.

Character layout: 6 slots of 50 mm each, separated by 40 mm gaps.
This ensures 20 mm wide navigable corridors between every character
even at the default clearance of 10 mm (5 mm radius + 5 mm user).

Characters occupy y in [85, 165]. Free strips:
    bottom: y in [10, 75]   (65 mm)
    top:    y in [175, 240] (65 mm)
    gaps between chars: 20 mm corridors at mid-map x values

Algorithm: backward A* seeded at the goal; heuristic = Euclidean
distance to the start. Stops when a node within GOAL_THRESH of
the start is popped. Parent pointers trace back to start->goal order.

Output: output_path.mp4 (exploration arrows then optimal path arrows)

Dependencies: numpy, opencv-python
    pip install numpy opencv-python
"""

import heapq
import math
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Map dimensions and algorithm constants
# ---------------------------------------------------------------------------

MAP_W = 600
MAP_H = 250
VIZ_SCALE = 3

XY_THRESH = 0.5
TH_THRESH = 30
N_THETA   = int(360 // TH_THRESH)   # 12 bins

VIX_MAX = int(MAP_W / XY_THRESH)    # 1200
VIY_MAX = int(MAP_H / XY_THRESH)    # 500

GOAL_THRESH = 1.5

# BGR colors 
COL_FREE    = (255, 255, 255)
COL_OBS     = (  0,   0,   0)   # black obstacles on white background
COL_START   = (197,   0,   0)   # blue filled circle
COL_GOAL    = (197,   0,   0)   # blue filled circle
COL_EXPLORE = (  0,  99,   0)   # dark green exploration arrows
COL_PATH    = (  0,   0, 197)   # red optimal path arrows


# ---------------------------------------------------------------------------
# Action set  (project spec Step 01)
#
# Each action turns by delta degrees then steps forward by L:
#   x'     = x + L * cos(theta + delta)
#   y'     = y + L * sin(theta + delta)
#   theta' = (theta + delta) mod 360
# ---------------------------------------------------------------------------

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

def _stroke_hit(x, y, xl, xh, yl, yh, c):
    return (xl - c) <= x <= (xh + c) and (yl - c) <= y <= (yh + c)


def is_obstacle(x, y, c):
    """
    Return True if (x, y) is inside an obstacle or its clearance zone.

    Checks map border walls first, then each rectangular stroke of SM7687.
    c = robot_radius + user_clearance.
    """
    if x < c or x > MAP_W - c or y < c or y > MAP_H - c:
        return True
    for (xl, xh, yl, yh) in _STROKES:
        if _stroke_hit(x, y, xl, xh, yl, yh, c):
            return True
    return False


def build_obstacle_grid(c):
    """
    Pre-compute a boolean occupancy grid at XY_THRESH resolution.
    Shape: (VIY_MAX, VIX_MAX) = (500, 1200). True means blocked.

    Parameters
    ----------
    c : total clearance (robot_radius + user_clearance)
    """
    xs = np.arange(VIX_MAX) * XY_THRESH
    ys = np.arange(VIY_MAX) * XY_THRESH
    X, Y = np.meshgrid(xs, ys)

    mask = (X < c) | (X > MAP_W - c) | (Y < c) | (Y > MAP_H - c)
    for (xl, xh, yl, yh) in _STROKES:
        mask |= (X >= xl - c) & (X <= xh + c) & (Y >= yl - c) & (Y <= yh + c)

    return mask.astype(np.uint8)


def build_map_image(c):
    """
    Build a BGR OpenCV canvas showing obstacles and clearance zones.
    Row 0 = top of image (OpenCV); Cartesian y increases upward.

    Parameters
    ----------
    c : total clearance (robot_radius + user_clearance)
    """
    cols = np.arange(MAP_W, dtype=np.float32)
    ycar = MAP_H - 1 - np.arange(MAP_H, dtype=np.float32)
    X, Y = np.meshgrid(cols, ycar)

    border = (X < c) | (X > MAP_W - c) | (Y < c) | (Y > MAP_H - c)

    obs_mask   = np.zeros((MAP_H, MAP_W), dtype=bool)
    clear_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
    for (xl, xh, yl, yh) in _STROKES:
        obs_mask   |= (X >= xl) & (X <= xh) & (Y >= yl) & (Y <= yh)
        clear_mask |= (X >= xl - c) & (X <= xh + c) & (Y >= yl - c) & (Y <= yh + c)

    canvas = np.full((MAP_H, MAP_W, 3), COL_FREE, dtype=np.uint8)
    canvas[obs_mask | border] = COL_OBS   # black obstacles only, no clearance tint
    return canvas


# ---------------------------------------------------------------------------
# Fast O(1) obstacle lookup using pre-computed grid
# ---------------------------------------------------------------------------

def obs_fast(x, y, grid):
    iy = int(round(y / XY_THRESH))
    ix = int(round(x / XY_THRESH))
    if ix < 0 or ix >= VIX_MAX or iy < 0 or iy >= VIY_MAX:
        return True
    return bool(grid[iy, ix])


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------

def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ---------------------------------------------------------------------------
# Segment collision check
#
# Samples the straight line from (x1,y1) to (x2,y2) at XY_THRESH
# intervals. Returns False if any sample is inside an obstacle.
# Prevents tunnelling through thin strokes when only endpoints are tested.
#
# Parameters
# ----------
# x1, y1, x2, y2 : segment endpoints (Cartesian mm)
# grid            : pre-computed occupancy grid
# ---------------------------------------------------------------------------

def segment_is_free(x1, y1, x2, y2, grid):
    dist    = euclidean(x1, y1, x2, y2)
    samples = max(1, int(math.ceil(dist / XY_THRESH)))
    for i in range(1, samples + 1):
        t   = i / samples
        xs  = x1 + t * (x2 - x1)
        ys  = y1 + t * (y2 - y1)
        rix = int(round(xs / XY_THRESH))
        riy = int(round(ys / XY_THRESH))
        if rix < 0 or rix >= grid.shape[1] or riy < 0 or riy >= grid.shape[0]:
            return False
        if grid[riy, rix]:
            return False
    return True


# ---------------------------------------------------------------------------
# Neighbour generator
#
# Applies all 5 actions and keeps only collision-free successors.
#
# Parameters
# ----------
# x, y, theta : current robot state
# L           : step size
# grid        : pre-computed occupancy grid
#
# Returns list of (nx, ny, ntheta, cost). Cost = L for every action.
# ---------------------------------------------------------------------------

def get_neighbours(x, y, theta, L, grid):
    result = []
    for act in ALL_ACTIONS:
        nx, ny, nt = act(x, y, theta, L)
        if not obs_fast(nx, ny, grid) and segment_is_free(x, y, nx, ny, grid):
            result.append((nx, ny, nt, float(L)))
    return result


# ---------------------------------------------------------------------------
# Visited matrix indexing  (project spec Step 03)
#
# Bins continuous (x, y, theta) into the 3-D visited array
# V[VIX_MAX x VIY_MAX x N_THETA] = V[1200 x 500 x 12].
#
# Spec examples:
#   (3.2,  4.7,  0) -> V[6][9][0]
#   (10.2, 8.8, 30) -> V[20][18][1]
# ---------------------------------------------------------------------------

def visited_idx(x, y, theta):
    ix = int(round(x / XY_THRESH))
    iy = int(round(y / XY_THRESH))
    it = int(round(theta / TH_THRESH)) % N_THETA
    ix = max(0, min(VIX_MAX - 1, ix))
    iy = max(0, min(VIY_MAX - 1, iy))
    return ix, iy, it


# ---------------------------------------------------------------------------
# Goal proximity check
# ---------------------------------------------------------------------------

def within_threshold(x, y, tx, ty):
    return euclidean(x, y, tx, ty) <= GOAL_THRESH


# ---------------------------------------------------------------------------
# Backward A* search  (project spec Step 03)
#
# Seeds the open list with the GOAL node (cost-to-come = 0).
# Heuristic h(n) = Euclidean distance from n to the START node.
# Terminates when a node within GOAL_THRESH of START is popped.
#
# Because the search runs goal->start, following parent pointers
# back from final_node produces the path in start->goal order.
#
# Parameters
# ----------
# start     : (x, y, theta) start pose
# goal      : (x, y, theta) goal pose
# step_size : action step length L (1-10)
# grid      : pre-computed occupancy grid
#
# Returns
# -------
# path           : list of (x, y, theta) waypoints in start->goal order,
#                  or None if no path exists
# explored_edges : list of ((x1,y1),(x2,y2)) pairs for visualization
# ---------------------------------------------------------------------------

def backward_astar(start, goal, step_size, grid):
    sx, sy, stheta = start
    gx, gy, gtheta = goal

    visited = np.zeros((VIX_MAX, VIY_MAX, N_THETA), dtype=np.uint8)
    parent  = {}
    cost_g  = {}
    counter = 0

    gk          = visited_idx(gx, gy, gtheta)
    cost_g[gk]  = 0.0
    parent[gk]  = (None, (gx, gy, gtheta))
    heap        = [(euclidean(gx, gy, sx, sy), counter, gx, gy, gtheta)]
    counter    += 1

    if within_threshold(gx, gy, sx, sy):
        return [(sx, sy, stheta), (gx, gy, gtheta)], []

    explored_edges = []
    final_node     = None

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

        for nx, ny, ntheta, move_cost in get_neighbours(cx, cy, ctheta, step_size, grid):
            nk = visited_idx(nx, ny, ntheta)
            nix, niy, nit = nk

            if visited[nix, niy, nit]:
                continue

            new_g = cost_g[ck] + move_cost
            if new_g < cost_g.get(nk, float("inf")):
                cost_g[nk] = new_g
                parent[nk] = (ck, (nx, ny, ntheta))
                explored_edges.append(((cx, cy), (nx, ny)))

                if within_threshold(nx, ny, sx, sy):
                    final_node = (nx, ny, ntheta)
                    break

                h = euclidean(nx, ny, sx, sy)
                heapq.heappush(heap, (new_g + h, counter, nx, ny, ntheta))
                counter += 1

        if final_node is not None:
            break

    if final_node is None:
        return None, explored_edges

    # Trace parent links from final_node back to the goal seed.
    # Search ran goal->start so this gives start->goal order.
    path = []
    cur  = visited_idx(*final_node)
    while cur is not None:
        pk, xyz = parent[cur]
        path.append(xyz)
        cur = pk

    fx, fy, _ = path[0]
    if abs(fx - sx) > 1e-6 or abs(fy - sy) > 1e-6:
        path.insert(0, (sx, sy, stheta))

    return path, explored_edges


# ---------------------------------------------------------------------------
# Visualization  (project spec Step 05)
#
# Each expanded edge and each path step is drawn as a directed arrow
# (vector), which satisfies the spec requirement to display the search
# tree as vectors. cv2.arrowedLine is equivalent to the matplotlib
# quiver approach shown in the spec sample code.
#
# Phase 1 draws exploration arrows (cyan) after search completes.
# Phase 2 overlays optimal path arrows (red) on the same canvas.
# ---------------------------------------------------------------------------

def cart_to_img(x, y):
    """Cartesian (x, y) -> OpenCV pixel (col, row)."""
    return int(round(x)), MAP_H - 1 - int(round(y))


def draw_arrow(canvas, x1, y1, x2, y2, color, thickness=1):
    p1 = cart_to_img(x1, y1)
    p2 = cart_to_img(x2, y2)
    if p1 != p2:
        cv2.arrowedLine(canvas, p1, p2, color, thickness, tipLength=0.4)


def visualise(canvas_base, start, goal, explored_edges, path,
              out_path="output_path.mp4"):
    """
    Write exploration and path animation to an mp4 file.

    Style matches the provided example:
      - 30 fps, 3x scale, black obstacles on white background
      - Dark green arrows for exploration, batched to ~15s total duration
      - Red arrows for path, one waypoint per frame
      - Blue filled circles for start and goal markers

    Parameters
    ----------
    canvas_base    : BGR map image (H x W x 3)
    start, goal    : (x, y, theta) tuples
    explored_edges : list of ((x1,y1),(x2,y2)) from backward_astar
    path           : list of (x, y, theta) waypoints in start->goal order
    out_path       : output filename
    """
    h, w   = canvas_base.shape[:2]
    out_h  = h * VIZ_SCALE
    out_w  = w * VIZ_SCALE
    fps    = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    canvas = canvas_base.copy()
    sx, sy, _ = start
    gx, gy, _ = goal

    # small filled circles for start and goal (radius 6 on base canvas -> 18px at 3x)
    cv2.circle(canvas, cart_to_img(sx, sy), 6, COL_START, -1)
    cv2.circle(canvas, cart_to_img(gx, gy), 6, COL_GOAL,  -1)

    def write_frame():
        frame = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        writer.write(frame)

    # phase 1: exploration
    # batch to ~450 output frames (15s at 30fps) regardless of edge count
    n     = len(explored_edges)
    batch = max(1, n // 450) if n else 1
    print(f"  Animating {n:,} exploration edges (batch={batch}, ~15s) ...")
    for i, ((x1, y1), (x2, y2)) in enumerate(explored_edges):
        draw_arrow(canvas, x1, y1, x2, y2, COL_EXPLORE, thickness=1)
        if i % batch == 0:
            write_frame()

    # 1s pause after exploration completes
    for _ in range(fps):
        write_frame()

    # phase 2: path - one waypoint per frame, matches example exactly
    print(f"  Animating {len(path)} path waypoints (1 per frame) ...")
    for i in range(1, len(path)):
        x1, y1, _ = path[i - 1]
        x2, y2, _ = path[i]
        draw_arrow(canvas, x1, y1, x2, y2, COL_PATH, thickness=1)
        write_frame()

    # 2s hold on final frame
    for _ in range(fps * 2):
        write_frame()

    writer.release()
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# User input helpers
# ---------------------------------------------------------------------------

def ask_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def ask_int_range(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Must be an integer between {lo} and {hi}.")
        except ValueError:
            print("  Please enter an integer.")


def ask_theta(prompt):
    """Accept any integer multiple of 30 and normalize to [0, 360)."""
    while True:
        try:
            v = int(input(prompt))
            if v % 30 == 0:
                return float(v % 360)
            print("  Theta must be a multiple of 30 (e.g. -60, 0, 30, 90 ...).")
        except ValueError:
            print("  Please enter an integer.")


def print_free_space_hint(c):
    """Print coordinates that are guaranteed to be in free space."""
    print("\n  Free space guide (with current clearance):")
    print(f"    Bottom strip : y in [10, {int(85 - c)}]  at any x in [10, 590]")
    print(f"    Top strip    : y in [{int(165 + c)}, 240] at any x in [10, 590]")
    print( "    Mid-map gaps : x ~ 100, 190, 280, 370, 460  (between characters)")
    print( "  Example inputs: (50, 50), (550, 50), (50, 200), (550, 200)")


def ask_point(label, c):
    """
    Prompt for (x, y, theta) and re-ask until the point is in free space.

    Parameters
    ----------
    label : display label shown to the user
    c     : total clearance for obstacle check
    """
    while True:
        print(f"\n  -- {label} --")
        x     = ask_float(f"    x     (0 - {MAP_W}): ")
        y     = ask_float(f"    y     (0 - {MAP_H}): ")
        theta = ask_theta( "    theta (multiple of 30 deg): ")
        if not (0 <= x <= MAP_W and 0 <= y <= MAP_H):
            print("  Outside map bounds. Try again.")
            continue
        if is_obstacle(x, y, c):
            print(f"  ({x}, {y}) is inside an obstacle or clearance zone.")
            print_free_space_hint(c)
            continue
        print(f"  {label} accepted: ({x}, {y}, {theta} deg)")
        return float(x), float(y), theta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nENPM661 Project 3 Phase 1 - Backward A* Path Planner")
    print("Team: Syed Ahmed, Sidharth Mathur, Jigar Shah\n")

    robot_r = ask_float("Robot radius   (mm) [default 5]: ", default=5.0)
    user_c  = ask_float("User clearance (mm) [default 5]: ", default=5.0)
    total_c = robot_r + user_c
    print(f"Total clearance: {total_c} mm")

    step_L  = ask_int_range("Step size L (1 - 10): ", 1, 10)

    print("\nCoordinates: Cartesian, origin at bottom-left corner.")
    print_free_space_hint(total_c)

    start = ask_point("Start", total_c)
    goal  = ask_point("Goal",  total_c)

    print("\nBuilding obstacle map ...")
    t0     = time.time()
    grid   = build_obstacle_grid(total_c)
    canvas = build_map_image(total_c)
    print(f"Map built in {time.time() - t0:.3f}s")

    print("\nRunning backward A* ...")
    t1 = time.time()
    path, explored = backward_astar(start, goal, step_L, grid)
    dt = time.time() - t1
    print(f"Search done in {dt:.3f}s  |  edges explored: {len(explored):,}")

    if path is None:
        print("\nNo path found. Try different start/goal or reduce clearance.")
        return

    cost = sum(
        euclidean(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        for i in range(len(path) - 1)
    )
    print(f"Path: {len(path)} waypoints, cost: {cost:.2f} units")

    print("\nGenerating animation ...")
    visualise(canvas, start, goal, explored, path)

    print(f"\nTotal runtime: {time.time() - t0:.2f}s")
    print("Output: output_path.mp4\n")


if __name__ == "__main__":
    main()