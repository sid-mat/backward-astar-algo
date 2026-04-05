# ENPM661 Project 3 Phase 1 - Backward A* Path Planner

## Team Members

| Name | Directory ID | UID |
|------|-------------|-----|
| Syed Yashal Ahmed | sahmed43 | 122288393 |
| Sidharth Mathur | sidmat03 | 122277687 |
| Jigar Shah | jshah310 | 121355690 |

---

## Description

Implements **backward A\*** search to find an optimal path for a circular
mobile robot on a 600 x 250 mm map. The obstacle space spells **SM7687**,
built from rectangular strokes defined using half-plane intersections
(semi-algebraic model). Each stroke boundary is inflated by the total
clearance c = robot radius + user clearance.

### Obstacle Layout

| Character | x slot (mm) | y range (mm) |
|-----------|-------------|--------------|
| S | 30 - 80 | 85 - 165 |
| M | 120 - 170 | 85 - 165 |
| 7 | 210 - 260 | 85 - 165 |
| 6 | 300 - 350 | 85 - 165 |
| 8 | 390 - 440 | 85 - 165 |
| 7 | 480 - 530 | 85 - 165 |

Characters are separated by 40 mm gaps. At default clearance (10 mm total),
each gap leaves a 20 mm navigable corridor.

### Free Space Guide (default clearance 10 mm)

- Bottom strip: y in [10, 75] at any x in [10, 590]
- Top strip: y in [175, 240] at any x in [10, 590]
- Mid-map corridors between characters: x ~ 100, 190, 280, 370, 460

---

## Dependencies

```
numpy
opencv-python
```

```bash
pip install numpy opencv-python
```

---

## How to Run

```bash
python a_star_syed_sidharth_jigar.py
```

The program will prompt you for:

1. **Robot radius** (mm) - default 5
2. **User clearance** (mm) - default 5
3. **Step size L** - integer from 1 to 10
4. **Start coordinates** `(x, y, theta)` - Cartesian, origin at bottom-left; theta in degrees (multiple of 30)
5. **Goal coordinates** `(x, y, theta)` - same format

If a point falls inside an obstacle or clearance zone, the program will
reject it with a message and suggest valid coordinates to try.

### Example Inputs

```
Robot radius   : 5
User clearance : 5
Step size L    : 5

Start x     : 50
Start y     : 50
Start theta : 0

Goal x      : 550
Goal y      : 200
Goal theta  : 0
```

### Coordinate System

- Origin: **bottom-left** corner of the map
- x increases to the **right** (0 to 600)
- y increases **upward** (0 to 250)
- theta is measured counter-clockwise from the positive x-axis
- theta must be a multiple of 30 (e.g. -60, -30, 0, 30, 60, 90)

---

## Output

- **`output_path.mp4`** - animation video showing:
  - Phase 1: Node exploration (dark green arrows)
  - Phase 2: Optimal path (red arrows)
  - Start and goal marked as filled blue circles

---

## Algorithm Details

| Parameter | Value |
|-----------|-------|
| Action set | 5 moves: delta-theta in {-60, -30, 0, +30, +60} degrees |
| Step cost | Euclidean distance = L |
| Heuristic | Euclidean distance to start node (backward search) |
| xy threshold | 0.5 unit |
| theta threshold | 30 degrees |
| Visited matrix | 1200 x 500 x 12 |
| Goal threshold | 1.5 unit radius |
| Video scale | 3x (output 1800 x 750 px) |
| Video fps | 30 |

### Why Backward?

The search seeds the open list at the **goal** and expands toward the **start**.
The heuristic at each node is its Euclidean distance to the start, so the
search is guided in the right direction. Following parent pointers from the
terminal node back to the goal seed naturally reconstructs the path in
start to goal order.

---

## File Structure

```
Proj3_phase1_Syed_Sidharth_Jigar.zip
├── a_star_syed_sidharth_jigar.py   <- main source code
├── README.md                       <- this file
└── output_path.mp4                 <- animation video
Proj3_Syed_Sidharth_Jigar.pdf       <- source code PDF (submitted separately)
```
