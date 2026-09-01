# Yuki Tool Dispatcher — Final Phase 1 Model Comparison

Date: 2026-08-19

## Outcome

Eight local models were tested one at a time against the same 18 real Yuki tools. No single model wins every dimension.

- **Best efficiency:** Hammer 2.1 1.5B
- **Best stable 3B balance:** Hammer 2.1 3B
- **Best hard-prompt strict accuracy:** xLAM-2 3B
- **Best hard-prompt speed/semantic balance:** Arch-Agent 3B
- **Not competitive:** Granite 4.1 3B, Arch-Function 3B, original xLAM 1B

The strongest practical default remains Hammer 3B if approximately 12-second broad latency is acceptable. Hammer 1.5B remains the most attractive deployment model when latency and memory matter. xLAM-2 3B and Arch-Agent 3B deserve Phase 2 evaluation before a final model is selected.

## Experimental contract

Every accepted run used:

- The 18 live Yuki schemas from the existing registry
- Model-native output format as the primary mode
- One stateless dispatcher generation per request
- No conversation history or previous tool result
- No assistant answer generation
- Model-only routing; no regex auto-detection or fallback
- 256-token ceiling with immediate complete-JSON stopping
- Strict schema validation before execution
- No tool execution during routing benchmarks
- Strict and execution-equivalent argument scoring reported separately

All converted model weights and Hugging Face source caches are on `/Volumes/madisk`. The internal disk contains only small compatibility links for older paths.

## Broad 72-case all-tools comparison

The harness verified identical cases across all eight reports.

| Rank | Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1= | Hammer 2.1 3B | **97.2%** | **91.7%** | **93.1%** | **98.6%** | **0.0%** | 11.92 s | 9.64 | 6.651 GB |
| 1= | xLAM-2 3B | **97.2%** | **91.7%** | 91.7% | 97.2% | **0.0%** | 16.13 s | 9.58 | 6.652 GB |
| 3 | Arch-Agent 3B | **97.2%** | 90.3% | **93.1%** | **98.6%** | 1.4% | 12.72 s | 9.73 | 6.652 GB |
| 4 | Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | **0.0%** | **4.69 s** | **18.47** | 3.624 GB |
| 5 | xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | **0.0%** | 6.22 s | 18.43 | 3.624 GB |
| 6 | Arch-Function 3B | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% | 13.58 s | 9.50 | 6.652 GB |
| 7 | Granite 4.1 3B | 91.7% | 73.6% | 75.0% | 90.3% | **0.0%** | 15.71 s | 8.61 | 7.441 GB |
| 8 | xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | **0.0%** | 5.65 s | 18.39 | 3.594 GB |

Broad strict exact-call ranking is based on the reported metric. Hammer 3B and xLAM-2 3B are tied numerically; the comparison file lists Hammer first because of input ordering.

## Targeted 96-case failure suite

The two earlier 1B xLAM reports were not run on this later targeted dataset, so the verified targeted comparison contains six models.

| Rank | Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | xLAM-2 3B | **99.0%** | **83.3%** | **86.5%** | **99.0%** | **0.0%** | 14.22 s | 9.11 | 6.652 GB |
| 2 | Arch-Agent 3B | 97.9% | 82.3% | **86.5%** | **99.0%** | 1.0% | 11.98 s | 9.54 | 6.652 GB |
| 3 | Hammer 2.1 3B | 87.5% | 80.2% | 81.3% | 87.5% | **0.0%** | 10.95 s | 9.41 | 6.651 GB |
| 4 | Arch-Function 3B | 89.6% | 78.1% | 79.2% | 81.3% | 10.4% | 13.81 s | 9.36 | 6.652 GB |
| 5 | Hammer 2.1 1.5B | 75.0% | 62.5% | 66.7% | 80.2% | **0.0%** | **4.64 s** | **18.44** | 3.624 GB |
| 6 | Granite 4.1 3B | 93.8% | 59.4% | 60.4% | 80.2% | 1.0% | 14.70 s | 8.53 | 7.441 GB |

## What the two suites mean together

The broad suite is easier and balanced across all 18 tools. It shows normal coverage and catches obvious ontology problems. The targeted suite intentionally concentrates on known failure families, so it is the better indicator of robustness but not a replacement for a larger representative dataset.

### Hammer 2.1 1.5B

Hammer 1.5B is the clear efficiency winner. It uses about 3.62 GB peak MLX memory, runs at about 18.5 generated tokens/second, and averages under five seconds. Its 87.5% broad strict accuracy is strong for its size.

Its weakness is difficult all-18-schema competition. Oracle subset exposure improved targeted strict exact calls from 62.5% to 71.9% with logical domains and 76.0% with focused confusion subsets. That is meaningful, but it did not recover into the 85–90% range hypothesized for a schema-overload-only explanation. Both capacity and schema competition matter.

### Hammer 2.1 3B

Hammer 3B is the safest current default. It leads or ties the broad suite, produced zero malformed output in both suites, and is the fastest of the strongest 3B options. Oracle subsets raised its targeted strict accuracy from 80.2% to 85.4% with logical domains and 87.5% with focused subsets.

Its targeted tool-selection score is lower than xLAM-2 3B and Arch-Agent, so difficult wording remains its main weakness.

### xLAM-2 3B

xLAM-2 3B is the hard-prompt accuracy leader. It selected 95 of 96 targeted tools correctly and produced no malformed output. It is the best candidate when selection robustness is more important than latency.

Its cost is speed: 16.13 seconds broad and 14.22 seconds targeted, despite using the same memory class as the other 3B models.

### Arch-Agent 3B

Arch-Agent is the strongest semantic balance. It ties xLAM-2 3B on targeted execution-equivalent accuracy and Hammer 3B on broad execution-equivalent accuracy, while running faster than xLAM-2 3B.

It rewrites arguments more often than xLAM-2 3B and produced one conversational refusal in each suite. Its Katanemo/DigitalOcean license also requires separate commercial permission, which may affect deployment independently of technical quality.

### Models to remove from the active shortlist

- **Arch-Function 3B:** too many malformed or conversational outputs for its speed and memory.
- **Granite 4.1 3B:** poor schema adherence, especially `read.filepath`; slowest generation and largest memory use. Apache 2.0 makes it interesting only as a future fine-tuning base.
- **Original xLAM 1B:** argument accuracy is far below the newer checkpoints.
- **xLAM-2 1B:** competent, but Hammer 1.5B is faster and substantially more accurate at essentially the same measured memory.

## Recommended shortlist and next experiment

Keep four checkpoints for now:

1. Hammer 2.1 1.5B — efficiency baseline
2. Hammer 2.1 3B — stable accuracy baseline
3. xLAM-2 3B — targeted robustness leader
4. Arch-Agent 3B — semantic/speed contender, subject to license suitability

Before a 300–500+ Phase 2 reliability study, run the realistic schema-selection experiment that was intentionally deferred:

```text
automatic domain selection
        ↓
selected Yuki schema subset
        ↓
one dispatcher generation
```

Measure end-to-end domain-selection errors separately from dispatcher errors. Hammer 1.5B is the key model for this test because oracle subsets helped substantially, but not enough to prove that schema overload is its only limitation.

Then run the 300–500+ Phase 2 dataset only on the surviving two or three models. The current 72-case broad suite is explicitly a smoke test and should not be used alone to lock a production model.

## Artifacts

- `final-eight-model-broad-comparison.json` — verified eight-model broad comparison
- `final-six-model-targeted-comparison.json` — verified six-model targeted comparison
- `ORACLE-SUBSET-REPORT.md` — Hammer 1.5B versus 3B oracle subset experiment
- Per-model Markdown reports and raw JSON/JSONL files are stored beside this report

