# Building Notes

## Strategy
1. Use get_task() to get target positions, compute bottom_z = z - sz/2 for each part
2. Determine dependencies: part j supports part i if j.top_z ~ i.bottom_z (tol ~1e-4) AND footprints overlap (strict interior, hi-lo > 1e-6 in BOTH x and y)
3. Topological sort (Kahn's) gives valid build order
4. Place parts bottom-up in dependency order, batching all parts of a level in parallel

## Key Lessons
- **planning_cpu_budget failures**: Common and transient. Almost always clears on a single retry after 1-2 other successful moves. Never retry the exact same call in a row without a fresh move first — batching a level then retrying the failed one works reliably.
- **Parallel move_part calls**: Batch all same-level (independent) parts together. Works well. One will often fail with planning_cpu_budget; retry it next turn.
- **Placement accuracy**: Typically < 0.5 mm, well within the 5 mm tolerance. A few land ~0.5 mm off (larger/longer parts) — still fine.
- The build order from topological sort is sufficient; no extra optimization needed.

## Tips
- If planner keeps failing on a part, place OTHER ready parts first to give it a "fresh start"
- Table at z=0; part rests on table if bottom_z ~ 0
- Edge-touching (sharing a face edge) does NOT count as support — need actual 2D area overlap
- After placing all parts, observe() to confirm all upright + errors < 5 mm before submit()
