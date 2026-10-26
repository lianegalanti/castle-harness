# Building Task Notes

## Strategy
- Always build bottom-up: place base parts first, then stack upward.
- Support check: a part can only be placed if something supports it (table or already-placed parts).
- `move_part` is reliable for placing parts at target positions; it handles pick/place in one call.
- The planner is smart about collision-free paths, so build order is the main concern.

## Tips
- If a part is on top of others, you can't place it before its supports are in place.
- observe() after each move to confirm placement (especially if something unexpected happens).
- Position errors of ~0.03-0.1mm are typical and well within the 5mm tolerance.
- All parts stay upright (yaw 0) so only center position matters.

## Common structures
- Pillars on table → wide plate spanning them → small cubes on plate → another plate → top piece.
- Build order follows the dependency graph: support parts first.
