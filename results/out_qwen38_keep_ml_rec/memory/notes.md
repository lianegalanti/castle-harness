# Build task learnings

## Approach that works
1. `get_task()` + `observe()` at start.
2. Compute build order by support dependency: a part "rests on" a part below it when
   top_z(lower) ≈ bottom_z(upper) (tol ~5mm) AND footprint overlap > ~30% of upper's area.
   A simple topological sort (place part when all its supporters are placed) gives a valid order.
   In practice the task part indices were already in a valid bottom-up order, but compute it to be safe.
3. Place parts one by one in that order with `move_part(part, x, y, z)` at exact target center, yaw 0.

## Key gotcha: planner_cpu_budget is TRANSIENT
- Many moves fail with reason "planning_cpu_budget" then `sim_steps: 0` (nothing moved, part stays parked).
- This is NOT a hard failure — just retry the same move. It eventually succeeds (took 4-12 tries for long/thin bridge
  parts and tall top-of-tower parts; small base parts usually 1-2).
- Failed moves do NOT disturb other placed parts (sim_steps 0 = no motion).
- Long thin bridges (0.18m x 0.02) and tall narrow top pieces hit the budget most often.
- Strategy on a failure streak: retry a few times; if still failing, do OTHER independent parts first, then return.

## Accuracy
- Final placement errors were 0.01–0.55mm, well under the 5mm limit. No re-placement needed.
- Parts that sit on other parts settle a hair low in z (gravity), but stays within tolerance.

## Submit check
- `observe()` all parts within 5mm & upright, then submit. Structure stayed stable.
