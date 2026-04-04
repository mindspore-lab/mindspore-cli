# Hotspot Prioritization

Read this file when `hotspot_summary.json` or `hotspot_summary.md` already
exists in the profiling output directory.

## Goal

Turn the msprof hotspot list into an optimization queue. Do not spread
attention evenly across all operators.

## Default Rule

Start from top 1 to top 3 operators by total time share.

- top 1: primary optimization target
- top 2: secondary target if top 1 is blocked or already optimized
- top 3: only discuss if it is still materially large or belongs to a different
  bottleneck family that may dominate after the first fix

Do not spend equal effort on low-share operators.

## How to Explain Priority

For each prioritized operator, say:

1. why it is high priority
2. what bottleneck family it belongs to
3. what the first optimization direction should be
4. what evidence should improve after rerun

## Default Optimization Direction by Category

### communication

Typical first directions:

- communication overlap
- bucketization or fusion
- reducing communication frequency
- removing unnecessary synchronization near the hotspot

Expected evidence to improve:

- collective time share
- collective count
- step tail

### computation_or_other

Typical first directions:

- operator fusion
- backend kernel path improvement
- graph/build or launch-path cleanup if the operator is actually a symptom of
  fragmented execution
- reducing redundant compute around the hotspot

Expected evidence to improve:

- hotspot operator time share
- end-to-end step time or latency

## Output Pattern

Use a compact priority list like:

1. operator name
   - share and why it is first
   - first optimization direction
   - rerun metric to watch
2. operator name
3. operator name

If only one operator clearly dominates, say so and avoid pretending the rest
are equally important.
