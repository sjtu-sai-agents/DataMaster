# TODO

## Post-run tree analysis

- Add a run analysis pass after new benchmark results are available.
- For the best-result node, summarize:
  - node id, benchmark, metric, depth, parent path, and visit count
  - tree structure statistics: total nodes, max depth, per-level width, red node count, black node count, buggy node count, evaluated black node count
  - data used by the best node: manifest, train pack source datasets, source ratios, raw/filtered source distribution, sample_count, transform usage, fallback retry usage
  - training metadata: key hyperparameters and effective train sample count
  - memory usage: memory files involved on the best path, parent/sibling memory injection, and whether memory appears correlated with score improvement
  - path analysis from root to best node, including where score gains first appeared
- Compare best-node characteristics across runs and benchmarks when enough runs accumulate.
