# Hammer 2.0 7B direct native tool calling — full 450-case report

Date: 2026-08-25

## Executive result

Hammer 2.0 7B produced the highest strict exact-call score among the three completed full-450 architectures in this comparison:

- **368/450 strict exact calls (81.8%)**
- **412/450 correct tool selections (91.6%)**
- **430/450 schema-valid outputs (95.6%)**
- **zero malformed outputs**

This is 10 more strict exact calls than the frozen Qwen3.8-27B → Hammer 2.1 1.5B pipeline and 25 more than GPT-OSS-20B direct native calling.

The gain is not free. Hammer 2.0 7B required **8.91 GB peak MLX memory** and averaged **6.24 seconds per decision with a verified static-prefix cache**. It is 1.35 times slower and uses 2.57 times the local dispatcher memory of the Qwen → Hammer 1.5B pipeline. GPT-OSS direct remained 8.4 times faster remotely.

No Yuki tool was executed.

## Model and runtime

- Source checkpoint: [`MadeAgents/Hammer2.0-7b`](https://huggingface.co/MadeAgents/Hammer2.0-7b)
- Frozen source revision: `2e267cc26e2185192e2de6d079f37bb878898640`
- Runtime checkpoint: `/Volumes/madisk/yuki-tool-dispatcher/Hammer2.0-7b-8bit`
- Runtime checkpoint size: 7.6 GB
- Quantization: MLX affine 8-bit, group size 64
- Base architecture reported by the checkpoint: Qwen2, 32K context
- Machine: MacBook Air M1, 16 GB unified memory

The official BF16 checkpoint is approximately 15.2 GB before runtime and KV overhead, so it was not loaded on this 16 GB machine. The 8-bit conversion preserves substantially more precision than a 4-bit test while fitting safely.

The model and Hugging Face cache were both stored on the external `madisk` volume. No internal model installation was created.

## Architecture tested

```text
raw user request
  -> Hammer 2.0 7B 8-bit MLX
       all 18 real Yuki Hammer-format schemas
       native Hammer JSON-array contract
  -> parser
  -> strict schema validation
  -> strict + execution-equivalent scoring
  -> stop
```

This run did not use Qwen, semantic delegation, Hammer 1.5B/3B, regex routing, a deterministic binder, conversation history, retries, voting, or any Yuki tool implementation.

## Frozen controls

- Dataset: all 450 frozen Phase 2 cases, 25 per tool.
- Dataset SHA-256: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`.
- Hammer schema SHA-256: `d545fa871c4802dfb08ba4b0fd3d0b82cb5d2c4685e85b636ec769bd460abfe7`.
- All 18 schemas were exposed on every request.
- One stateless greedy generation per case.
- Generation ceiling: 256 tokens, with immediate complete-JSON stopping.
- Empty arrays were parsed as explicit Hammer rejections, not malformed JSON.
- Execution attempts: zero.

## Static-prefix cache validation

Uncached all-18-schema smoke latency was 34.83 seconds per case because each request repeatedly prefills roughly 2.2K prompt tokens through an 8B model.

The production-style benchmark used an immutable 2,048-token cache containing only the fixed tool schemas, task instruction, and format instruction. It contained no user request and no previous model output.

The cache was accepted only after a second 18-case smoke produced byte-for-byte identical raw generations to the uncached smoke for all 18 tools.

| Smoke metric | Uncached | Cached |
|---|---:|---:|
| Tool selection | 100% | 100% |
| Strict exact | 88.9% | 88.9% |
| Schema valid | 100% | 100% |
| Malformed | 0% | 0% |
| Average latency | 34.83 s | 5.68 s |
| Peak MLX memory | 8.91 GB | 8.91 GB |

The full run's one-time static prefix build took 28.02 seconds. Model load and prefix-build time are not included in per-case latency.

## Full-450 metrics

| Metric | Hammer 2.0 7B direct |
|---|---:|
| Tool selection | **91.6% (412/450)** |
| Strict argument accuracy | 83.1% (374/450) |
| Execution-equivalent argument accuracy | 83.1% (374/450) |
| Strict exact call | **81.8% (368/450)** |
| Execution-equivalent exact call | **81.8% (368/450)** |
| Strict exact when tool was correct | 89.3% |
| Schema-valid output | **95.6% (430/450)** |
| Explicit rejection | 4.4% (20/450) |
| Malformed output | **0.0% (0/450)** |
| Average cached latency | 6.236 s |
| p50 cached latency | 6.130 s |
| p95 cached latency | 8.181 s |
| Average generation rate | 7.53 tok/s |
| Average full prompt length | 2,210.9 tokens |
| Average generated length | 19.61 tokens |
| Peak MLX memory | 8.91 GB |
| API cost | $0 |

Strict and execution-equivalent scores are identical in this arm. None of the mistakes qualified as a harmless normalization under the frozen per-tool rules.

## Architecture comparison

These arms use the same 450 raw source cases and gold calls, but the architectures differ: Hammer 7B and GPT-OSS receive raw requests directly with all 18 schemas, whereas Qwen → Hammer uses semantic delegation and selected domain schemas.

| Metric | Hammer 2.0 7B direct | GPT-OSS-20B direct | Qwen3.8 → Hammer 1.5B |
|---|---:|---:|---:|
| Tool selection | 91.6% | **94.9%** | 90.7% |
| Strict exact call | **81.8%** | 76.2% | 79.6% |
| Execution-equivalent exact | **81.8%** | 77.8% | 79.8% |
| Schema valid | 95.6% | **98.7%** | 92.4% |
| Malformed | **0.0%** | 1.1% | 0.4% |
| Average latency | 6.236 s cached | **0.741 s** | 4.631 s sequential |
| Generation rate | 7.53 tok/s | **55.10 tok/s** | 19.39 tok/s for Hammer |
| Local peak model memory | 8.91 GB | remote | **3.47 GB Hammer peak** |
| Measured API cost | **$0** | $0.0259 | $0.1331 |

Hammer 7B versus GPT-OSS:

- 15 fewer correct tool choices.
- 25 more strict exact calls.
- 14 fewer schema-valid outputs.
- 61 cases were fixed only by Hammer 7B, while 36 were correct only for GPT-OSS.

Hammer 7B versus Qwen → Hammer 1.5B:

- 4 more correct tool choices.
- 10 more strict exact calls.
- 14 more schema-valid outputs.
- 44 cases were fixed only by Hammer 7B, while 34 were correct only for Qwen → Hammer.

## Per-tool strict comparison

Each row contains 25 cases.

| Tool | H20 tool | H20 strict | GPT strict | Q→H strict |
|---|---:|---:|---:|---:|
| `time` | 96% | 96% | 84% | **100%** |
| `weather` | 100% | 100% | 100% | 100% |
| `fetch` | 100% | **100%** | 100% | 92% |
| `search` | 88% | **64%** | 64% | 60% |
| `calc` | 100% | **100%** | 76% | 96% |
| `hardware` | 96% | **96%** | 88% | 84% |
| `see` | 88% | **72%** | 60% | 36% |
| `image` | 100% | **60%** | 8% | 48% |
| `read` | 88% | 88% | 56% | **96%** |
| `shell` | 100% | **100%** | 96% | 88% |
| `yuki_write` | 76% | 60% | **92%** | 80% |
| `yuki_read` | 60% | 60% | 84% | **96%** |
| `yuki_list` | 84% | 84% | 96% | **100%** |
| `yuki_delete` | 92% | **92%** | 88% | 92% |
| `yuki_append` | 96% | 80% | 68% | **84%** |
| `remember` | 88% | 68% | **92%** | 76% |
| `recall` | 100% | **88%** | 84% | 44% |
| `ask_claude` | 96% | **64%** | 36% | 60% |

## What the 7B capacity fixed

### `see`

Hammer 7B selected `see` in 22/25 cases and reached 18/25 strict exact calls. This is substantially above the frozen Qwen → Hammer 1.5B result of 10/25 selections and 9/25 strict calls.

Only one `see` case was explicitly rejected. The remaining misses were one `read`, one `search`, and four correct-tool argument errors. Larger capacity materially helps the visual-inspection concept.

### `recall`

Hammer 7B selected `recall` in all 25 cases, rejected none, and reached 22/25 strict exact calls. This effectively removes the old Hammer 1.5B recall rejection cluster.

### Exact payloads

Hammer 7B was much stronger than GPT-OSS on the most difficult regenerated strings:

- `image`: 60% strict versus 8%.
- `read`: 88% versus 56%.
- `ask_claude`: 64% versus 36%.
- `see`: 72% versus 60%.

The model still rewrote ten image prompts and eight correctly routed Claude questions, so deterministic source binding remains relevant.

## What the 7B capacity did not fix

### Explicit rejection

Hammer returned a valid native empty array on 20 tool-required cases. These are not malformed generations, but they are failed dispatches and therefore schema-invalid for this benchmark.

The rejection distribution was:

- `yuki_list`: 4
- `read`: 3
- `remember`: 3
- `search`: 2
- `yuki_read`: 2
- `yuki_delete`: 2
- one each for `time`, `see`, `yuki_append`, and `ask_claude`

Preservation cases were especially rejection-prone: 7/48 (14.6%).

### Normal filesystem versus Yuki filesystem

The strongest remaining ontology failure is `read` versus `yuki_read`:

- Eight expected `yuki_read` cases were routed to normal `read`.
- Two more `yuki_read` cases were rejected.
- `yuki_read` selection and strict accuracy were only 60%.

This is worse than both GPT-OSS direct and Qwen → Hammer. More capacity did not solve this distinction under all-18 exposure.

### Write versus append

Six expected `yuki_write` cases were routed to `yuki_append`. `yuki_write` tool selection fell to 76% and strict accuracy to 60%.

This is a real operation-ontology weakness, not merely a copied-string error.

## Failure decomposition

There were 82 strict failures:

| Primary class | Cases |
|---|---:|
| Explicit native `[]` rejection | 20 |
| Wrong non-reject tool | 18 |
| Correct tool, wrong argument | 44 |
| Malformed output | 0 |
| Safe normalization only | 0 |

The most important clean tool confusions were:

- `yuki_read` → `read`: 8
- `yuki_write` → `yuki_append`: 6
- `see` → `read` or `search`: 2
- `search` → `recall`: 1
- `hardware` → `shell`: 1

## Dataset-category results

| Category | Tool selection | Strict exact | Schema valid | Reject |
|---|---:|---:|---:|---:|
| Normal | 95.0% | 89.1% | 99.0% | 1.0% |
| Paraphrase | 90.9% | 80.0% | 93.6% | 6.4% |
| Confusion | 91.7% | 77.8% | 95.8% | 4.2% |
| Preservation | 79.2% | 64.6% | 85.4% | 14.6% |
| Adversarial | 88.9% | 72.2% | 94.4% | 5.6% |

## Decision

Hammer 2.0 7B demonstrates a real capacity benefit. It is the first compared arm to exceed 80% strict exact accuracy on the frozen full 450 without a deterministic binder, and it materially fixes `see`, `recall`, image prompts, and Claude payloads.

It is not an obvious default dispatcher for the M1 Air:

- The gain over Qwen3.8 → Hammer 1.5B is only 10 additional strict calls.
- Cached latency is 1.35 times higher than the complete sequential Qwen → Hammer path.
- Local peak memory is 2.57 times the Hammer 1.5B peak.
- Twenty valid tool-required requests were still explicitly rejected.
- The Yuki filesystem ontology regressed badly.

The model is credible for an **offline, local, accuracy-first** Yuki configuration. For the fastest architecture, GPT-OSS direct remains far ahead. For the most memory-efficient local dispatcher, Hammer 1.5B remains more attractive.

The most informative next step is not another broad model run. Apply the frozen deterministic source binder to Hammer 7B's correct tool selections and separately test a narrow ontology clarification for:

- `read` versus `yuki_read`
- `yuki_write` versus `yuki_append`
- valid tool-required requests currently rejected as `[]`

That will show whether Hammer 7B's 91.6% selection can be converted into a substantially higher end-to-end exact score without another model generation pass.

## Verification

- 450/450 unique checkpoint records.
- 82 strict-failure JSONL records.
- Raw report, checkpoint, and failure checksums verified.
- 178 dispatcher tests passed.
- Ruff lint passed.
- Model process exited after the run.
- Yuki tool execution attempts: zero.
