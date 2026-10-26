# Box-building task notes

## Reading the 3 views
- top: x horiz, y vert (looking down -z)
- front: x horiz, z vert (looking +y)
- side: y horiz, z vert (looking -x)
Use part sizes to snap: on table z=size_z/2; on others bottom=top of support. Touching parts share exact edge/centerline.

## Generic approach that works
1. Build a consistent 3D model: assign each part (x,y,z center).
2. Verify no volume overlaps; each non-floor part's bottom rests on table or a placed part.
3. Build order bottom-up (floor first, then supports, then what rests on them).
4. move_part in that order; observe after surprises; submit.

## Tolerance 5mm. Grid reading ~ few mm; snap to size/edges to make exact.
