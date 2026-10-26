# Block-building task notes (UR5e + parallel gripper, box parts)

## Core mechanics
- move_part(part,x,y,z): picks part, places center at (x,y,z), arm returns home. Returns ok + final_pose.
- Positions are part CENTERS. Upright parts: table-resting z = size_z/2. Stacked: bottom = top of support.
- get_task gives part sizes + 3 orthographic images (top x/y, front x/z, side y/z). Target coords NOT given — infer from images using metric grids + part sizes for exactness.

## CRITICAL LESSONS
1. **NEVER make parallel (multiple) move_part calls in one block.** A parallel move caused a part to fall over (upright=false) and left the arm stuck mid-grasp (gripper closed, not at home). Always one move at a time.
2. **`planning_cpu_budget` failures are TRANSIENT** — just retry the same move. Often succeeds on 2nd/3rd try. Long parts time out more often; interleave with other parts and retry.
3. **`no_feasible_removal_or_grasp` on an already-placed/stuck part** = it's on its side or blocked; the gripper can't grab non-upright parts. Hard to recover without reset_scene (which resets everything).
4. **`recovery_state_not_safe`** = the move disturbed something (a part fell). Observe immediately.
5. **`no_feasible_removal_or_grasp` on the part you're moving** can also mean the planner can't find a path. Retry or change approach.
6. Once you place a part, it's HARD to move again (planner may refuse if it's buried). Build order matters: put base + supports first, stack on top. A buried part is a trap.
7. Planner slows dramatically as scene clutters (wall_s grew from 16s to 700s+ per move). Budget is consumed fast. Keep moves efficient.

## Strategy that works
- Read 3 views, build a self-consistent (x,y,z) per part: z from stacking chain, x from top/front, y from top/side.
- Build bottom-up: base slab first, then supports on it, then stacks. Vertical cube chains are easy & reliable (small parts, clear support).
- Verify each part's bottom overlaps its support's top in x AND y before relying on it.
- The planner's support check is a good feedback signal: if it rejects with a support reason, your z or support assumption is wrong.

## Reading orthographic views
- Top view: horizontal=x, vertical=y (looking down -z).
- Front view: horizontal=x, vertical=z (looking +y).
- Side view: horizontal=y, vertical=z (looking -x).
- Overlapping/occluded parts are the hard part. Distinguish front-vs-back in y from side view, and near-vs-far in x from front view.
- Exact positions: touching parts share edge or centerline; use sizes to snap to multiples of 0.005.

## This run's result
Could not fully complete a 31-part structure: coordinate inference from views was too uncertain for the dense middle, a part (3) fell and got stuck, planner flakiness ate budget. Placed base + right tower correctly. Partial score.
