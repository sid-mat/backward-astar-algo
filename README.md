# ENPM661 Project 3 Phase 1 — Backward A* Path Planner

## Team Members
| Name | Directory ID | UID |
|------|-------------|-----|
| [Name 1] | [dir_id_1] | [uid_1] |
| [Name 2] | [dir_id_2] | [uid_2] |

**GitHub Repository:** [Your GitHub Link Here]
*(Must have at least 5 commits with meaningful messages)*

---

## Description

Implements the **backward A\*** search algorithm to find an optimal path for a
circular mobile robot on a 600 × 250 mm map with three obstacles:

| Obstacle | Description |
|----------|-------------|
| Rectangle 1 | x ∈ [150, 175], y ∈ [0, 125] |
| Rectangle 2 | x ∈ [250, 275], y ∈ [125, 250] |
| Circle | centre (400, 110), radius 50 |

The robot has configurable radius and clearance (default 5 mm each).

---

## Dependencies

```
numpy
opencv-python
```

Install with:

```bash
pip install numpy opencv-python
```

---

## How to Run

```bash
python a_star_solution.py
```

The program will prompt you for:

1. **Robot radius** (mm) — default 5
2. **User clearance** (mm) — default 5
3. **Step size L** — integer from 1 to 10
4. **Start coordinates** `(x, y, θ)` — Cartesian, origin at bottom-left; θ in degrees (multiple of 30)
5. **Goal coordinates** `(x, y, θ)` — same format

### Example Inputs

```
Robot radius   : 5
User clearance : 5
Step size L    : 5

Start x  : 50
Start y  : 50
Start θ  : 0

Goal x   : 550
Goal y   : 200
Goal θ   : 0
```

### Coordinate System

- Origin: **bottom-left** corner of the map
- x increases to the **right** (0 → 600)
- y increases **upward** (0 → 250)
- θ is measured **counter-clockwise** from the positive x-axis (East)
- θ must be a multiple of 30° (e.g. −60°, −30°, 0°, 30°, 60°, 90°, …)

---

## Output

- **`output_path.mp4`** — Video showing:
  - Phase 1: Node exploration (cyan arrows)
  - Phase 2: Optimal path (red arrows)
- Final frame shown in an OpenCV window (press any key to close)

---

## Algorithm Details

| Parameter | Value |
|-----------|-------|
| Action set | 5 moves: Δθ ∈ {−60°, −30°, 0°, +30°, +60°} from current heading |
| Step cost | Euclidean distance = L |
| Heuristic | Euclidean distance to **start** (backward search) |
| xy threshold | 0.5 unit |
| θ threshold | 30° |
| Visited matrix | 1200 × 500 × 12 |
| Goal threshold | 1.5 unit radius |

### Why Backward?

The search is seeded at the **goal** and expands toward the **start**.
The heuristic at each node estimates distance to the start, guiding the search
efficiently. Backtracking parent pointers produces the forward start → goal path.

---

## File Structure

```
Proj3_phase1_firstname1_firstname2.zip
├── a_star_solution.py      ← main source code
├── README.md               ← this file
└── output_path.mp4         ← animation video
Proj3_firstname1_firstname2.pdf   ← source code PDF (submitted separately)
```
