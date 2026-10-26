# UR5e box-building notes

## Core loop that works
1. get_task(); read memory first.
2. Compute, for each part: bottom = z - sz/2, top = z + sz/2, x/y extents.
3. Build "supports" map: part B rests on table if bottom~0; on part A if A.top ~ B.bottom AND xy overlap (gap>1e-4).
4. Topological sort (treat 'table' as pre-done). This gives a valid build order.
5. move_part each part to its target_position (yaw 0). observe() periodically.

## Key gotchas
- `move_part` failures with reason `planning_cpu_budget` have sim_steps=0 -> nothing moved, scene NOT disturbed. Just RETRY.
- Parallel tool calls DO execute in the order listed (confirmed). So batching a dependency-respecting sequence is fine.
- BUT some parts (long thin beams bridging small point-supports, or small posts in a dense area) can hit planning_cpu_budget repeatedly even on solo retry when the scene is CROWDED. FIX: place such tricky pieces EARLY, right after their supports, while few neighbors are down.
- After placement parts settle with tiny yaw/orientation_error (< ~0.3 deg) and ~0.1mm position error -- fine, well under 5mm.
- Supports must be physically in place before a part (planner rejects otherwise). Respect topo order.
- If the scene gets ruined -> reset_scene() (does NOT refund budget).

## Useful python
box extents: (px-sx/2, px+sx/2, py-sy/2, py+sy/2, pz-sz/2, pz+sz/2)
overlap(a,b): (min(a[1],b[1])-max(a[0],b[0]))>1e-4 and (min(a[3],b[3])-max(a[2],b[2]))>1e-4
