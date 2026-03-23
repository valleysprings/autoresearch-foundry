# LiveCodeBench v6

This task mirrors the full `livecodebench/code_generation_lite` `release_v6` problem set as a dataset-task under the existing single-file evolve runtime.

It does not depend on the official LiveCodeBench harness. Instead:

- `prepare.py` lazily materializes only the requested prefix of items into a local manifest plus per-item cached problem files
- `verifier.py` runs each item directly as either:
  - a `stdin` script benchmark, or
  - a LeetCode-style functional benchmark via `class Solution`

The full release contains `1055` problems. The demo UI warns against running more than `50` items at once because the first run also builds the local cache for the requested prefix.
