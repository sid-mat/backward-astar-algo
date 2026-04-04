"""
actions.py — ENPM661 Project 03 Phase 1
----------------------------------------
Contains the 5 action functions for the differential-drive robot, plus the
helper utilities (euclidean distance, segment collision check, neighbour
generator) that depend on them.

Each action rotates the robot by a fixed delta, then moves it forward by
step size L:
    x' = x + L * cos(θ + Δθ)
    y' = y + L * sin(θ + Δθ)
    θ' = (θ + Δθ) mod 360

The 5 actions and their deltas:
    action_straight  →  Δθ =   0°
    action_left30    →  Δθ = +30°
    action_right30   →  Δθ = -30°
    action_left60    →  Δθ = +60°
    action_right60   →  Δθ = -60°
"""

import math


# ── Constants (must stay in sync with main file) ───────────────────────────────
XY_THRESH = 0.5   # spatial threshold used by segment_is_free sampling


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPER — shared motion equation used by all 5 actions
# ═══════════════════════════════════════════════════════════════════════════════

def _apply(x: float, y: float, theta: float, delta: float, L: float):
    """
    Apply a turn of `delta` degrees then move forward by `L` units.

    Parameters
    ----------
    x, y   : current position (mm)
    theta  : current heading (degrees, 0–360)
    delta  : heading change for this action (degrees)
    L      : step size (units)

    Returns
    -------
    (nx, ny, new_theta) — new position rounded to 2 decimal places,
                          new heading in [0, 360).
    """
    new_theta = (theta + delta) % 360
    rad = math.radians(new_theta)
    nx = x + L * math.cos(rad)
    ny = y + L * math.sin(rad)
    return round(nx, 2), round(ny, 2), new_theta


# ═══════════════════════════════════════════════════════════════════════════════
#  THE 5 ACTION FUNCTIONS  (project spec §Step 01)
# ═══════════════════════════════════════════════════════════════════════════════

def action_straight(x: float, y: float, theta: float, L: float):
    """Move straight ahead — no heading change (Δθ = 0°)."""
    return _apply(x, y, theta, 0, L)


def action_left30(x: float, y: float, theta: float, L: float):
    """Turn left 30°, then move forward (Δθ = +30°)."""
    return _apply(x, y, theta, 30, L)


def action_right30(x: float, y: float, theta: float, L: float):
    """Turn right 30°, then move forward (Δθ = -30°)."""
    return _apply(x, y, theta, -30, L)


def action_left60(x: float, y: float, theta: float, L: float):
    """Turn left 60°, then move forward (Δθ = +60°)."""
    return _apply(x, y, theta, 60, L)


def action_right60(x: float, y: float, theta: float, L: float):
    """Turn right 60°, then move forward (Δθ = -60°)."""
    return _apply(x, y, theta, -60, L)


# List used by get_neighbours to iterate all actions in one loop
ALL_ACTIONS = [
    action_straight,
    action_left30,
    action_right30,
    action_left60,
    action_right60,
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def segment_is_free(x1: float, y1: float, x2: float, y2: float,
                    obs_grid, sample_step: float = XY_THRESH) -> bool:
    """
    Walk along the straight line from (x1,y1) to (x2,y2) in steps of
    `sample_step` and return False if any sampled point is inside an obstacle.

    This prevents the robot from 'tunnelling' through a thin obstacle when
    only the endpoint is checked.
    """
    dist = euclidean(x1, y1, x2, y2)
    samples = max(1, int(math.ceil(dist / sample_step)))
    for i in range(1, samples + 1):
        t = i / samples
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        if obs_grid[int(round(ys / sample_step)),
                    int(round(xs / sample_step))]:
            return False
    return True


def get_neighbours(x: float, y: float, theta: float, L: float,
                   obs_grid, obs_fast_fn) -> list:
    """
    Apply all 5 actions to the current state and return only the valid ones.

    A neighbour is valid when:
      1. Its endpoint is not inside an obstacle (obs_fast_fn check).
      2. The straight-line segment to it is fully collision-free (segment_is_free).

    Parameters
    ----------
    x, y, theta : current robot state
    L           : step size
    obs_grid    : pre-computed boolean grid (VIY_MAX × VIX_MAX)
    obs_fast_fn : the obs_fast() function from the main file

    Returns
    -------
    List of (nx, ny, new_theta, cost) tuples for each valid action.
    Cost of every action equals L.
    """
    result = []
    for act in ALL_ACTIONS:
        nx, ny, nt = act(x, y, theta, L)
        if not obs_fast_fn(nx, ny, obs_grid) and \
           segment_is_free(x, y, nx, ny, obs_grid):
            result.append((nx, ny, nt, float(L)))
    return result