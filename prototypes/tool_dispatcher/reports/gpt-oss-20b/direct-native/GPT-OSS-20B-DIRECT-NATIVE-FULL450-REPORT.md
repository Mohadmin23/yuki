# GPT-OSS-20B direct native tool calling — full 450-case report

Date: 2026-08-25

## Executive result

Using GPT-OSS-20B as the final function dispatcher works much better than using it as Yuki's six-domain semantic main brain.

With all 18 real Yuki schemas exposed on every request, GPT-OSS selected the correct final tool in **427/450 cases (94.9%)**. That exceeds the frozen Qwen3.8-27B → Hammer 1.5B pipeline's **408/450 (90.7%)**.

The remaining weakness is argument extraction. GPT-OSS produced **343/450 strict exact calls (76.2%)**, below Qwen → Hammer's **358/450 (79.6%)**. Under execution-equivalent scoring, GPT-OSS reached **350/450 (77.8%)** versus **359/450 (79.8%)**.

The architecture tradeoff is therefore clear:

- GPT-OSS direct is better at choosing the final tool.
- Qwen → Hammer remains slightly better at producing the complete exact call.
- GPT-OSS direct is approximately 6.25 times faster than the measured sequential Qwen + Hammer path and costs approximately one fifth as much in API spend.

No Yuki tool was executed.

## Architecture tested

```text
raw user request
  -> GPT-OSS-20B
       all 18 strict OpenAI-compatible Yuki schemas
       native API tool call required
  -> normalize native tool_calls
  -> strict parse
  -> strict schema validation
  -> deterministic scoring
  -> stop
```

This test did not use:

- Qwen
- Hammer
- semantic delegation
- automatic domain routing
- regex routing
- deterministic source binding
- conversation history
- retries, voting, or self-correction
- Yuki tool execution

## Frozen controls

- Dataset: all 450 Phase 2 cases, 25 per Yuki tool.
- Dataset SHA-256: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`.
- Registry source: Yuki's live 18-tool registry.
- Strict adapted schema SHA-256: `22139a7c2a7ad583ec38b1fb98715e876c1a9b51554607859bde75688dc6523d`.
- Model: `openai/gpt-oss-20b` through OpenRouter.
- Provider: DeepInfra pinned, fallback disabled.
- Reasoning: mandatory low effort, excluded from visible output.
- Generation ceiling: 256 tokens.
- `tool_choice`: required.
- One stateless generation per request.
- Multiple calls rejected by the parser.
- Execution attempts: zero.

The provider did not accept the `parallel_tool_calls` transport parameter under strict capability filtering, so that parameter was omitted. Exactly-one-call behavior was still enforced by the prompt, required tool choice, and parser.

## Smoke gate

The final controlled 18-case smoke produced:

| Metric | Result |
|---|---:|
| Tool selection | 94.4% (17/18) |
| Strict exact call | 83.3% (15/18) |
| Schema valid | 94.4% |
| Malformed | 5.6% |
| Average latency | 0.742 s |
| Cost | $0.00103 |

The smoke was healthy enough to justify the full run.

## Full-450 metrics

| Metric | GPT-OSS direct |
|---|---:|
| Tool selection | **94.9% (427/450)** |
| Strict argument accuracy | 77.6% (349/450) |
| Execution-equivalent argument accuracy | 79.1% (356/450) |
| Strict exact call | **76.2% (343/450)** |
| Execution-equivalent exact call | **77.8% (350/450)** |
| Strict exact when tool was correct | 80.3% |
| Equivalent exact when tool was correct | 82.0% |
| Schema-valid output | **98.7% (444/450)** |
| Malformed output | 1.1% (5/450) |
| Model rejection | 0.0% |
| Average latency | 0.741 s |
| p50 latency | 0.618 s |
| p95 latency | 1.402 s |
| Average prompt tokens | 1,727.3 |
| Average generated tokens | 40.84 |
| Average generation rate | 55.10 tok/s |
| API cost | $0.02589 |

## Direct GPT-OSS versus Qwen → Hammer

| Metric | GPT-OSS direct | Qwen3.8 → Hammer 1.5B | Difference |
|---|---:|---:|---:|
| Tool selection | **94.9%** | 90.7% | **+4.2 pp** |
| Strict exact call | 76.2% | **79.6%** | -3.3 pp |
| Execution-equivalent exact | 77.8% | **79.8%** | -2.0 pp |
| Schema valid | **98.7%** | 92.4% | +6.2 pp |
| Malformed | 1.1% | **0.4%** | +0.7 pp |
| Average routing latency | **0.741 s** | 4.631 s sequential | 6.25x faster |
| API cost | **$0.0259** | $0.1331 | 5.14x cheaper |
| Local dispatcher memory | **none** | 3.473 GB MLX peak | remote versus local |

The Qwen + Hammer latency is the exact per-case sum of the frozen Qwen Stage A and Hammer Stage B generation latencies. It is not an estimate from unrelated averages.

GPT-OSS direct produced 19 more correct tool choices than Qwen → Hammer, but 15 fewer strict exact calls. Better selection is being lost to argument rewriting.

## Per-tool comparison

Each row contains 25 cases.

| Tool | GPT tool | Q→H tool | GPT strict | Q→H strict |
|---|---:|---:|---:|---:|
| `time` | 84% | **100%** | 84% | **100%** |
| `weather` | 100% | 100% | 100% | 100% |
| `fetch` | **100%** | 92% | **100%** | 92% |
| `search` | **96%** | 92% | **64%** | 60% |
| `calc` | 96% | **100%** | 76% | **96%** |
| `hardware` | **100%** | 92% | **88%** | 84% |
| `see` | **88%** | 40% | **60%** | 36% |
| `image` | **100%** | 88% | 8% | **48%** |
| `read` | 100% | 100% | 56% | **96%** |
| `shell` | **100%** | 96% | **96%** | 88% |
| `yuki_write` | **96%** | 92% | **92%** | 80% |
| `yuki_read` | 84% | **100%** | 84% | **96%** |
| `yuki_list` | 96% | **100%** | 96% | **100%** |
| `yuki_delete` | **100%** | 96% | 88% | **92%** |
| `yuki_append` | 76% | **96%** | 68% | **84%** |
| `remember` | 100% | 100% | **92%** | 76% |
| `recall` | **100%** | 52% | **84%** | 44% |
| `ask_claude` | 92% | **96%** | 36% | **60%** |

## What native calling fixed

### `see`

The old Qwen → Hammer pipeline selected `see` in only 40% of cases and reached 36% strict exact calls. GPT-OSS direct reached 88% selection and 60% strict exact.

This is strong evidence that the old `see` problem was primarily the semantic-delegation/Hammer interface rather than an inherently impossible tool ontology.

### `recall`

Hammer selected `recall` in only 52% of frozen cases. GPT-OSS direct selected it in all 25 cases and produced 84% strict exact calls.

Native direct calling effectively removes the previous Hammer rejection cluster.

### Schema validity

GPT-OSS direct produced schema-valid calls in 444 cases, compared with 416 for Qwen → Hammer. It did not produce any empty-array/model-reject output recognized by the parser.

## What native calling did not fix

### Image prompts

GPT-OSS selected `image` in all 25 cases but matched the frozen expected prompt in only 2 cases. It typically included surrounding instructions, negations, or full-sentence phrasing rather than the minimal gold prompt.

This is the single largest exact-argument failure family: 23 cases.

### Normal file paths

GPT-OSS selected `read` in all 25 cases but reached only 56% strict exact calls. Examples included dropping a leading slash:

```text
expected: /tmp/FooBar.JSON
actual:   tmp/FooBar.JSON
```

This is execution-breaking and must remain exact.

### External-agent questions

`ask_claude` selected correctly in 23/25 cases, but strict exact accuracy was only 36%. The model often answered, summarized, or rewrote the question payload instead of transporting the user's wording.

### Yuki append

`yuki_append` reached only 76% selection because GPT-OSS confused it with `yuki_write` five times. This remains a real operation-level ontology problem.

## Strict failure taxonomy

There were 107 strict failures:

| Primary category | Cases |
|---|---:|
| Wrong argument after correct tool | 77 |
| Wrong tool with otherwise parseable/valid output | 17 |
| Safe normalization only | 7 |
| Malformed output | 5 |
| Schema-invalid invented tool | 1 |

The five malformed outputs were one each for `time`, `search`, `calc`, `see`, and `yuki_list`. They were natural-language answers or JSON error objects instead of native tool calls. One additional `time` case invented a nonexistent `reject` function.

Among correct-tool argument failures, the largest groups were:

| Tool | Failures |
|---|---:|
| `image` | 23 |
| `ask_claude` | 14 strict failures after correct selection: 12 non-equivalent and 2 normalization-only |
| `read` | 11 |
| `search` | 8 |
| `see` | 7 |
| `calc` | 5 strict rewrites, mostly execution-equivalent |
| `recall` | 4 |

## Dataset-category comparison

| Category | GPT tool | Q→H tool | GPT strict | Q→H strict |
|---|---:|---:|---:|---:|
| Normal | **96.5%** | 92.6% | 79.2% | **85.1%** |
| Paraphrase | **94.5%** | 92.7% | 79.1% | **80.9%** |
| Confusion | **97.2%** | 83.3% | 69.4% | **77.8%** |
| Preservation | 85.4% | **91.7%** | **70.8%** | 64.6% |
| Adversarial | **94.4%** | 83.3% | **66.7%** | 55.6% |

GPT-OSS direct is more robust at selecting tools in confusion and adversarial language. Qwen → Hammer retains better exact-call performance on normal and confusion cases.

## Comparison with GPT-OSS semantic delegation

The earlier GPT-OSS semantic main-brain arm reached only 86.2% domain accuracy. Direct native tool selection reached 94.9% across the same 450 requests.

This shows that GPT-OSS's poor semantic-delegation result was partly an interface mismatch. The model performs much better when allowed to use its native function-calling behavior and choose the final function directly.

## Cost and performance

- Full native run cost: `$0.02589181`.
- Average latency: `0.741 s`.
- p95 latency: `1.402 s`.
- Average hidden reasoning: `11.33 tokens`.
- Maximum hidden reasoning: `144 tokens`.
- Local MLX model memory: none; inference was remote.
- Process RSS observed by the harness: at most 78.141 MB.

This does not show how a locally hosted GPT-OSS-20B would perform on the M1 Air. It measures the OpenRouter/DeepInfra deployment only.

## Decision

GPT-OSS direct native calling is a credible alternative architecture, unlike GPT-OSS semantic delegation.

It should not yet replace the Qwen → Hammer path based solely on strict accuracy: 76.2% remains below 79.6%. But its 94.9% tool selection, 98.7% schema validity, low latency, and low cost justify one focused follow-up before rejecting it.

The next experiment should not rerun GPT-OSS. Use these frozen 450 native calls and apply deterministic source binding to exact-sensitive arguments, starting with:

- `image.prompt`
- `read.filepath`
- `ask_claude.question`
- `search.query`
- `see.target`
- `recall.topic`
- exact Yuki filenames/content where applicable

If deterministic binding repairs at least 16 strict cases without breaking correct calls, GPT-OSS direct would exceed the frozen Qwen → Hammer strict total of 358/450. The 23 image failures, 14 ask-Claude failures, and 11 read failures provide enough headroom for that to be plausible.

## Artifacts

- Full raw report: `gpt-oss-20b-all18-native-full450-v1.json`
  - SHA-256: `17b7c1ef351dd7cff359189482d79d807c8b4a14a1443c08a818df1f9ca6d1d1`
- Failure JSONL: `gpt-oss-20b-all18-native-full450-v1-failures.jsonl`
  - SHA-256: `6cd54acf77e4d783a82fb542c449884ebf5ab751047884edb4eec2509dc5ad50`
- Controlled smoke: `gpt-oss-20b-all18-native-smoke18-v4.json`
  - SHA-256: `4049a7e9ff1eddf0ca335c1f3ac8e7474306f847de7c469fe60835c6cdd20685`
- Machine-readable comparison: `GPT-OSS-20B-DIRECT-NATIVE-FULL450-COMPARISON-v1.json`

No frozen source, Qwen report, Hammer report, production Yuki tool, or benchmark gold was modified.
