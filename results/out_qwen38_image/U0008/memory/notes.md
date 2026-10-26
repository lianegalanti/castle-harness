# Build task notes

## Coordinate inference
- All parts at y=0 typically (a wall in x-z plane). Read x from front/side views, z from front/side.
- A part on the table has z = size_z/2. A part on top of others: bottom = top of supports.
- "Touching parts share exact edges/centerlines" is the key to exact x. When a small block sits at an edge of a beam, do NOT assume its center aligns with the beam edge — it may be inset so its center of mass is supported.

## Stability / failure mode (important)
- A small cube placed with its center RIGHT at a support's edge tips over (CM at edge → torque). The move fails with "execution_guard:open:...:contacts=1" and the part ends up knocked down/off target.
- Fix: ensure the supported part's CENTER sits well inside the support's footprint. If a beam spans x=[a,b], a 0.02-wide block centered at x must satisfy x-0.01 >= a and x+0.01 <= b. When target shows a block near a beam end, recompute: block touches the NEXT part (e.g. a box on top) on its inner edge, so its center is inset by its half-width from that inner edge.
- After any failed move, observe(); the part may have fallen off. reset_scene() to start clean if disturbed.

## Build order
- Bottom-up: posts/base first, then lintels, then small blocks on top, then top beam, then top box.
- Beams (long in x) support both their own weight and the top box if box width <= beam length.
- place each part then move_part returns arm home; small settle (z drops <0.1mm) is normal and fine.

## Accuracy
- move_part places centers to ~0.01-0.1 mm of the given (x,y,z). Good enough for 5 mm tolerance.
- submit passes if every part within 5mm, upright, resting correctly, still for 1s.
