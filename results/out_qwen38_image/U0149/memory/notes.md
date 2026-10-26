# Build notes

- move_part returns final_pose with exact achieved position; error field is tiny (<100um). Trust it.
- Build bottom-up: feet/table first, then supports, then things resting on top. Order that avoids blocking.
- Compute target z from stack heights: part resting on table has z = size_z/2; on others, bottom = top of support, z = bottom + size_z/2.
- Use part sizes + symmetry from views to get exact x/y (e.g. centered at midpoint between two supports). Don't eyeball the grid.
- Side view (looking along -x) often reveals y=0 centering for a planar structure.
- observe() at end to confirm all upright & stable before submit.
