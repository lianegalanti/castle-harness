# Box-building task notes (UR5e)

## General method
- get_task gives part sizes + 3 orthographic views (top: x/y, front: x/z, side: y/z). Each view has metric grid.
- Read (x,y) from TOP view, read z-level from FRONT/SIDE (vertical axis). Compute exact z from support stacking: zc = bottom + size_z/2, bottom = top of supports.
- Build ORDER matters: place supports first (table-level parts), then things resting on them. Planner rejects moves with no support under destination or no collision-free path.
- Tolerance 5mm per part, upright, resting on correct support, stable 1s.
- observe() after unexpected results. reset_scene() if ruined (no budget refund).

## Part size classes
- posts 0.02x0.02x0.04: 0,1,2,3,8,9,21 (tall upright)
- thin slabs on table etc.
- big slabs: 13 (0.02x0.2x0.05), 15 (0.18x0.04x0.02), 17 (0.04x0.11x0.02), 18 (0.1x0.04x0.01), 22 (0.02x0.08x0.01)

## Gotchas
- A part labeled in 2 places in a view = misread; trust dimensions to disambiguate.
- Long-in-x parts (size x=0.18): 4,15. Long-in-y (size y=0.2): 6,7,10,13.

## This task: hard to read precisely; best-effort. Document final table below after solving.
