# Build Strategy Notes

## Key Principles
1. **Build bottom-to-top**: Always place parts that rest on the table first, then build up.
2. **Check support**: A part's bottom (z - size_z/2) must align with the top of whatever supports it.
3. **Build order = topological sort of the dependency graph**: Part A depends on Part B if A rests on B.
4. **All moves worked perfectly** when following this approach - no need for reset.

## move_part() behavior
- Returns accurate final_pose with sub-0.1mm error on this task.
- The planner handles path planning internally; just give it the target center.
- Failed moves can disturb other parts, so always observe() if anything goes wrong.

## Code pattern for determining build order
- For each part, compute bottom_z = target_z - size_z/2
- A part rests on the table if bottom_z ≈ 0
- A part rests on another part if its bottom_z ≈ that part's top_z
- Sort by bottom_z ascending; within same level, order doesn't matter
