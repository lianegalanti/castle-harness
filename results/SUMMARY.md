# ML comparison report

Metrics are read from each arm's `results.jsonl` and corresponding episode JSONL. Failed `move_part` counts and per-request input-token maxima are computed from episode tool results and `usage`; no estimates are used.

## Load-average observations

Per-run load averages are not stored in `results.jsonl` or episode JSONL, so they cannot be attributed exactly to individual tasks. The monitoring snapshots available from the runs were:

- 13:55 (-0400): 8.18 / 8.36 / 8.56
- 14:10 (-0400): 21.20 / 19.76 / 15.38
- 16:37 (-0400): 23.63 / 32.03 / 28.95

These are host-wide 1/5/15-minute load averages on 256 cores, not per-run averages.

## Baseline

| Task | Passed | Parts | Input tokens | Output tokens | wall_s | Tool calls | Failed move_part by reason | Max request input tokens | Video |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| U0002 | true | 19/19 | 239475 | 15116 | 3293.4311885619536 | 34 | planning_cpu_budget: 8 | 19899 | — |
| U0007 | true | 19/19 | 188586 | 10753 | 3814.807002417743 | 31 | planning_cpu_budget: 4 | 19247 | — |
| U0011 | true | 24/24 | 318000 | 10125 | 3395.654062792193 | 39 | planning_cpu_budget: 7 | 23986 | — |
| U0012 | true | 30/30 | 286958 | 12041 | 3947.1034080078825 | 41 | planning_cpu_budget: 4 | 25430 | — |
| U0024 | true | 29/29 | 841153 | 19632 | 2868.0586395012215 | 48 | planning_cpu_budget: 9 | 34838 | — |
| U0052 | true | 31/31 | 234811 | 9267 | 4424.351028226316 | 42 | planning_cpu_budget: 4 | 23805 | — |

## Worked

| Task | Passed | Parts | Input tokens | Output tokens | wall_s | Tool calls | Failed move_part by reason | Max request input tokens | Video |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| U0002 | true | 19/19 | 436301 | 28684 | 3302.71154520195 | 37 | planning_cpu_budget: 10 | 23527 | — |
| U0007 | true | 19/19 | 195622 | 17275 | 2431.8201879113913 | 30 | planning_cpu_budget: 4 | 20797 | — |
| U0011 | true | 24/24 | 439288 | 19706 | 4829.3151427432895 | 36 | planning_cpu_budget: 6 | 23021 | — |
| U0012 | false | 17/30 | 3942160 | 87907 | 6739.336094004102 | 142 | planning_cpu_budget: 57 | 79727 | — |
| U0024 | true | 29/29 | 795218 | 20063 | 3888.011826209724 | 47 | planning_cpu_budget: 8 | 29085 | — |
| U0052 | false | 1/31 | 3989339 | 75255 | 10186.601684364956 | 114 | planning_cpu_budget: 29 | 70644 | — |

## Recorded ML

| Run | Task | Passed | Parts | Input tokens | Output tokens | wall_s | Tool calls | Failed move_part by reason | Max request input tokens | Video |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| out_qwen38_rec_U0002 | U0002 | true | 19/19 | 521553 | 22882 | 7177.097374264151 | 38 | planning_cpu_budget: 11 | 23954 | videos/ml_v6/000_U0002.mp4 |
| out_qwen38_rec_U0007 | U0007 | true | 19/19 | 440100 | 17151 | 5387.110669190995 | 33 | planning_cpu_budget: 7 | 22858 | videos/ml_v6/001_U0007.mp4 |
| out_qwen38_rec_U0011 | U0011 | true | 24/24 | 458618 | 23668 | 7009.484393367078 | 34 | planning_cpu_budget: 4 | 23219 | videos/ml_v6/002_U0011.mp4 |
| out_qwen38_rec_U0012 | U0012 | true | 30/30 | 342243 | 13379 | 8391.178862646688 | 40 | planning_cpu_budget: 3 | 24583 | videos/ml_v6/003_U0012.mp4 |
| out_qwen38_keep_ml_rec | U0024 | true | 29/29 | 1286033 | 31031 | 5944.477834285703 | 61 | planning_cpu_budget: 23 | 36169 | videos/ml_v6/004_U0024.mp4 |
| out_qwen38_rec_U0052 | U0052 | true | 31/31 | 1151427 | 27404 | 8843.355123091023 | 52 | planning_cpu_budget: 7 | 37069 | videos/ml_v6/005_U0052.mp4 |

## Image-only

Per-part position-error summaries use the `position_error_m` values in `results.jsonl`, converted to mm. Threshold counts are `within 5 mm / 1 cm / 2 cm` out of the reported parts.

| Task | Passed | Parts | Input tokens | Output tokens | wall_s | Tool calls | Failed move_part by reason | Max request input tokens | Per-part position error (median/max; threshold counts) | Video |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| U0008 | false | 5/7 | 132412 | 28213 | 1741.3308419249952 | 19 | execution_guard:open:4:contacts=1: 1 | 12300 | 0.072/10.023 mm; 5/7 / 6/7 / 7/7 | videos/image_v1/000_U0008.mp4 |
| U0080 | true | 7/7 | 60937 | 10517 | 812.1003648098558 | 14 | — | 8264 | 4.947/4.998 mm; 7/7 / 7/7 / 7/7 | videos/image_v1/001_U0080.mp4 |
| U0149 | true | 7/7 | 49513 | 16631 | 1022.456279009115 | 13 | — | 7464 | 0.052/0.094 mm; 7/7 / 7/7 / 7/7 | videos/image_v1/002_U0149.mp4 |
| U0002 | false | 2/19 | 1479042 | 161231 | 4115.198330665007 | 70 | planning_cpu_budget: 44; no_feasible_removal_or_grasp: 3 | 39744 | 197.024/376.826 mm; 2/19 / 6/19 / 6/19 | videos/image_v1/003_U0002.mp4 |
| U0007 | false | 1/19 | 1167105 | 160657 | 5077.006193474866 | 69 | planning_cpu_budget: 31 | 33406 | 30.012/430.011 mm; 1/19 / 3/19 / 8/19 | videos/image_v1/004_U0007.mp4 |
| U0011 | false | 1/24 | 201142 | 91427 | 1754.5288856537081 | 22 | planning_cpu_budget: 9 | 17141 | 279.665/429.363 mm; 1/24 / 1/24 / 3/24 | videos/image_v1/005_U0011.mp4 |
| U0012 | false | 0/30 | 2146123 | 121481 | 11128.782650323119 | 213 | planning_cpu_budget: 150; no_feasible_removal_or_grasp: 2; execution_guard:open:2:contacts=1: 1; whole_scene_boundary: 1 | 103331 | 244.981/420.348 mm; 0/30 / 0/30 / 1/30 | Full-layout blocked (implicit terminal submit absent from JSONL); no-text package: `deliver/castle_videos/image_only/U0012/U0012.mp4` |
| U0024 | false | 2/29 | 3255767 | 383781 | 10874.093147476204 | 79 | planning_cpu_budget: 14; execution_guard:open:4:contacts=0: 1; recovery_state_not_safe: 5; no_feasible_removal_or_grasp: 1 | 85456 | 267.192/478.933 mm; 2/29 / 2/29 / 4/29 | Full-layout blocked (implicit terminal submit absent from JSONL); no-text package: `deliver/castle_videos/image_only/U0024/U0024.mp4` |
| U0052 | false | 0/31 | 1649341 | 143563 | 6480.979949765839 | 50 | planning_cpu_budget: 16; no_feasible_removal_or_grasp: 3; recovery_state_not_safe: 1 | 53877 | 233.297/421.224 mm; 0/31 / 1/31 / 1/31 | videos/image_v1/008_U0052.mp4 |

## Easy comparison: positions-mode vs image-only

Each task has one row per mode; the position-error summary and threshold counts are computed from each run's `per_part_errors`.

| Mode | Task | Passed | Parts | Input tokens | Output tokens | wall_s | Tool calls | Failed move_part by reason | Max request input tokens | Per-part position error (median/max; within 5 mm / 1 cm / 2 cm) | Video |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| Positions-mode (recorded) | U0008 | true | 7/7 | 41380 | 4114 | 1019.3680326701142 | 13 | — | 6524 | 0.051/0.099 mm; 7/7 / 7/7 / 7/7 | videos/easy_v6/000_U0008.mp4 |
| Image-only | U0008 | false | 5/7 | 132412 | 28213 | 1741.3308419249952 | 19 | execution_guard:open:4:contacts=1: 1 | 12300 | 0.072/10.023 mm; 5/7 / 6/7 / 7/7 | videos/image_v1/000_U0008.mp4 |
| Positions-mode (recorded) | U0080 | true | 7/7 | 56793 | 4594 | 1014.9889793409966 | 14 | — | 8175 | 0.166/0.235 mm; 7/7 / 7/7 / 7/7 | videos/easy_v6/001_U0080.mp4 |
| Image-only | U0080 | true | 7/7 | 60937 | 10517 | 812.1003648098558 | 14 | — | 8264 | 4.947/4.998 mm; 7/7 / 7/7 / 7/7 | videos/image_v1/001_U0080.mp4 |
| Positions-mode (recorded) | U0149 | true | 7/7 | 46547 | 4071 | 1116.8410445838235 | 13 | — | 6943 | 0.054/0.091 mm; 7/7 / 7/7 / 7/7 | videos/easy_v6/002_U0149.mp4 |
| Image-only | U0149 | true | 7/7 | 49513 | 16631 | 1022.456279009115 | 13 | — | 7464 | 0.052/0.094 mm; 7/7 / 7/7 / 7/7 | videos/image_v1/002_U0149.mp4 |

## Memory diffs

The initial `list_memory` result for both baseline and worked was `{'files': []}`; the starting `memory/notes.md` was empty/absent.

### Baseline: `memory/notes.md`

```diff
--- /dev/null
+++ memory/notes.md
@@ -0,0 +1,19 @@
+# Building Notes
+
+## Strategy
+1. Use get_task() to get target positions, compute bottom_z = z - sz/2 for each part
+2. Determine dependencies: part j supports part i if j.top_z ~ i.bottom_z (tol ~1e-4) AND footprints overlap (strict interior, hi-lo > 1e-6 in BOTH x and y)
+3. Topological sort (Kahn's) gives valid build order
+4. Place parts bottom-up in dependency order, batching all parts of a level in parallel
+
+## Key Lessons
+- **planning_cpu_budget failures**: Common and transient. Almost always clears on a single retry after 1-2 other successful moves. Never retry the exact same call in a row without a fresh move first — batching a level then retrying the failed one works reliably.
+- **Parallel move_part calls**: Batch all same-level (independent) parts together. Works well. One will often fail with planning_cpu_budget; retry it next turn.
+- **Placement accuracy**: Typically < 0.5 mm, well within the 5 mm tolerance. A few land ~0.5 mm off (larger/longer parts) — still fine.
+- The build order from topological sort is sufficient; no extra optimization needed.
+
+## Tips
+- If planner keeps failing on a part, place OTHER ready parts first to give it a "fresh start"
+- Table at z=0; part rests on table if bottom_z ~ 0
+- Edge-touching (sharing a face edge) does NOT count as support — need actual 2D area overlap
+- After placing all parts, observe() to confirm all upright + errors < 5 mm before submit()
```

### Worked: `memory/notes.md`

```diff
--- /dev/null
+++ memory/notes.md
@@ -0,0 +1,20 @@
+# UR5e box-building notes
+
+## Core loop that works
+1. get_task(); read memory first.
+2. Compute, for each part: bottom = z - sz/2, top = z + sz/2, x/y extents.
+3. Build "supports" map: part B rests on table if bottom~0; on part A if A.top ~ B.bottom AND xy overlap (gap>1e-4).
+4. Topological sort (treat 'table' as pre-done). This gives a valid build order.
+5. move_part each part to its target_position (yaw 0). observe() periodically.
+
+## Key gotchas
+- `move_part` failures with reason `planning_cpu_budget` have sim_steps=0 -> nothing moved, scene NOT disturbed. Just RETRY.
+- Parallel tool calls DO execute in the order listed (confirmed). So batching a dependency-respecting sequence is fine.
+- BUT some parts (long thin beams bridging small point-supports, or small posts in a dense area) can hit planning_cpu_budget repeatedly even on solo retry when the scene is CROWDED. FIX: place such tricky pieces EARLY, right after their supports, while few neighbors are down.
+- After placement parts settle with tiny yaw/orientation_error (< ~0.3 deg) and ~0.1mm position error -- fine, well under 5mm.
+- Supports must be physically in place before a part (planner rejects otherwise). Respect topo order.
+- If the scene gets ruined -> reset_scene() (does NOT refund budget).
+
+## Useful python
+box extents: (px-sx/2, px+sx/2, py-sy/2, py+sy/2, pz-sz/2, pz+sz/2)
+overlap(a,b): (min(a[1],b[1])-max(a[0],b[0]))>1e-4 and (min(a[3],b[3])-max(a[2],b[2]))>1e-4
```
