# Benchmark protocol

| Component | Setting |
| --- | --- |
| Target | Current-inclusive discounted cost-to-go, gamma 0.99 |
| Labels | Training-set q75/q90; sensitivity q80/q95 |
| Split | Reconstructed episode groups, 60/20/20; seeds 0, 1, 2 |
| Input | 32-step prefix and shape-matched current-only control |
| Sample caps | 100,000 training samples, 50,000 validation samples; uncapped test split |
| Preprocessing | Training-only thresholds and normalization |
| Models | GRU, FlatMLP, causal Transformer |
| Main learning rates | GRU 3e-4; FlatMLP and Transformer 3e-3, selected on validation data |
| Target-ablation learning rate | 1e-3 |
| Main endpoint | Step catastrophic AUPRC and matched empirical Step prevalence |
| Auxiliary endpoints | Outcome AUPRC, Step ECE, and lead-time summaries |

Each task is averaged over three split seeds before statistical comparisons. F1/F2/F3 use Holm families of 3/3/2, 20,000 task-bootstrap samples (seed 260922), and 100,000 paired sign flips (seed 260921). Target-ablation comparisons use separate F4/F5 families.

The target is defined retrospectively from recorded costs. Episode groups are disjoint across splits; source metadata does not establish disjoint data-collection runs. FISOR Point Robot uses signed `h` as specified by its publisher's loader, with upper-quantile labels under the same protocol. Alarm utility uses fixed-log replay, with thresholds selected on validation scores.

Recorded run manifests contain the exact seeds, architecture settings, label definitions, indices, and source hashes for each experiment. Historical schema identifiers are retained in these records to identify the recorded protocol version.
