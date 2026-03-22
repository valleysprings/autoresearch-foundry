# Benchmark Protocol

This repo now has two benchmark tiers:

- `easy-regression`: the original deterministic codegen tasks in `set-logic`, `counting`, and `numeric`
- `hard-math`: `count-primes-up-to`, `count-change-ways`, and `count-n-queens`

## Why this split exists

The easy tier is good for catching verifier or memory regressions, but many tasks saturate in generation 1. The hard-math tier is where frontier selection and memory architecture should prove they can keep evolving beyond the first accepted win.

## Primary experiment matrix

Run each task under these conditions:

1. `current baseline`: single-incumbent hill-climb, if you need a historical comparison from old artifacts
2. `frontier-memory`: current default engine in this repo
3. `frontier-no-memory`: same engine, but temporarily clear retrieved memory to measure memory value

Recommended budget:

- `generation_budget=20`
- `candidate_budget=3`
- repeat each task at least `3` times with fresh run directories

## Commands

Single task, 20 generations:

```bash
python3 -m app.entries.discrete_demo --task count-n-queens --generation-budget 20
```

Whole catalog, 20 generations:

```bash
python3 -m app.entries.discrete_demo --generation-budget 20
```

## Metrics to track

- `frontier_accepts`: number of generations accepted into the local frontier
- `global_best_improves`: number of generations that improved the overall best candidate
- `first_best_generation`: generation index of the first global-best improvement after generation 1
- `tail_stagnation`: generations since the last global-best improvement
- `write_backs`: total memory writes
- `failure_write_backs`: memory writes with `verifier_status in {fail,error}`
- `memory_pollution_rate`: passing-but-non-improving generations divided by failure write-backs
- `best_objective_gain`: winner objective minus baseline objective

## Success criteria

The frontier-memory engine should beat the older hill-climb pattern on the hard-math tier by:

- producing more than one accepted generation on a meaningful share of runs
- reducing passing-but-useless failure write-backs toward zero
- showing lower tail stagnation on `count-change-ways` and `count-n-queens`

## External references already cloned in this repo

- `external/openevolve`
  Use this as the reference for frontier/diversity ideas such as islands, archives, and parent sampling.
- `external/autoresearch`
  Use this as the reference for the keep-or-discard experiment discipline around a fixed evaluator budget.

## Next benchmarks for a v2 harness

These are worth cloning once the runtime supports natural-language problem statements instead of only Python function bodies:

- `MATH-500`
- `AIME-style olympiad sets`
- `miniF2F`
- OpenEvolve's circle-packing and geometric optimization examples
