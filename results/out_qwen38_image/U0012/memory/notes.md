UR5e box-building task notes (general):
- move_part uses RRT + a per-move CPU/wall budget (~24s). Stochastic: same move succeeds ~25-50%, fails = "planning_cpu_budget" (sim_steps=0, part UNCHANGED, safe to retry).
- KEY: congestion kills the planner. Parked parts form small stacked piles; only the TOP of each pile is graspable. A scene with TWO dense clusters (built target + staging pile) => ~0% success.
- BEST strategy: reset_scene to the sparse parked layout, then place each part DIRECTLY to its target in build order (no staging area). Keep the workspace sparse. Grab a pile's top to free the parts beneath it.
- Long thin plates (>=0.10 long, <=0.02 thick) are hardest to plan; retry them a lot.
- reset_scene gives a FRESH wall clock (great after burning the budget).
- Read targets: x from FRONT view, y from SIDE view, z from stacking (z=center; on table z=H/2; on others z=support_top+H/2). 5mm tolerance.
- Build bottom-up (table/legs first, then floor, then tower).
- Batch 8-10 move_part calls in parallel; expect a few to succeed each batch.
