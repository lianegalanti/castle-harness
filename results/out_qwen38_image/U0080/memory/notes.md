# Box-stacking task notes

## Reading targets from views
- Front view (x/z) gives x centers + z stacking order (bottom->top).
- Top view (x/y) gives y centers + which parts are hidden under larger ones.
- Side view (y/z) confirms y centers and stacking.
- All positions exact: table z=0 -> resting part center z = size_z/2; stacked part bottom = top of support.
- Symmetric towers: central parts share the same x (symmetry axis), legs symmetric about it.
- 5mm tolerance, so read to ~mm and refine with sizes.

## Build order
- Always bottom-up so each part has support when placed.
- Legs/base first, then bridge, then up the tower.

## Gotchas
- A part spanning two separate supports (bridge): COM must lie inside convex hull of
  contact patches, not just the union. Symmetric bridge = stable. Planner may still
  reject; observe reason.
- Overhanging slabs: stable if COM (center) projects over the smaller support below.
- Place small top parts last; arm approaches from above the narrow tower fine.
- move_part: pick->place->home. observe() after anything unexpected.
