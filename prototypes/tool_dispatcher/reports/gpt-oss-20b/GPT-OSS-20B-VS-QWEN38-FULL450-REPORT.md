# GPT-OSS-20B vs Qwen3.8-27B — Full 450-case Yuki comparison

Date: 2026-08-25

## Executive conclusion

`openai/gpt-oss-20b` is substantially faster and cheaper than `qwen/qwen3.8-27b`, but it is not a competitive replacement for Qwen as Yuki's semantic main brain under the current production-delegation contract.

Across the exact same frozen 450 requests:

- Qwen selected the correct schema domain in **446/450 cases (99.1%)**.
- GPT-OSS selected the correct schema domain in **388/450 cases (86.2%)**.
- Qwen preserved the complete required exact payload in **256/300 payload-bearing cases (85.3%)**.
- GPT-OSS preserved it in **220/300 cases (73.3%)**.
- Qwen produced the exact expected `verbatim` array in **251/300 payload-bearing cases (83.7%)**.
- GPT-OSS did so in **211/300 cases (70.3%)**.
- GPT-OSS averaged **1.032 seconds** per delegation versus **2.145 seconds** for Qwen.
- GPT-OSS cost **$0.0121** for 450 cases versus **$0.1331** for Qwen.

The domain result is decisive for the current architecture. Hammer receives only the schemas from the selected domain. With 62 wrong or missing GPT-OSS domains, the correct tool would be unavailable in those cases. Therefore, even with a theoretically perfect Hammer dispatcher, this GPT-OSS run is capped at **86.2% final tool-selection accuracy**. The already measured Qwen → Hammer 1.5B pipeline reached **90.7% tool selection**.

GPT-OSS may still deserve a separate test as a direct 18-tool dispatcher. This report tests it only as the semantic main brain replacing Qwen.

## Architecture tested

```text
raw user request
  -> main-brain semantic delegation JSON
       request
       domain_hint
       verbatim[]
  -> parse
  -> deterministic scoring
  -> stop
```

No Yuki tool was executed. No Hammer model was loaded for the GPT-OSS arm because the external volume containing Hammer 1.5B was not mounted.

For reference, the frozen Qwen downstream architecture was:

```text
raw user request
  -> Qwen3.8-27B semantic delegation
  -> selected domain schema subset
  -> Hammer 1.5B native tool call
  -> parse + validate + score
  -> stop
```

## Frozen data and controls

- Dataset: `production-delegation-source-v1`.
- Cases: 450, exactly 25 cases for each of Yuki's 18 tools.
- Derived dataset SHA-256: `ea6711ce0548d3771166f8f2f9a5e59ebd73acb14a9ddf7ecbf2f6589f1744c3`.
- Source Phase 2 SHA-256: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`.
- Delegation prompt SHA-256 for both arms: `1189b8cb529f8f13592e4952b9e1da31093592dfe2710c96796bc9bedb120cbb`.
- Same delegation schema, parser, domain labels, gold data, and deterministic scorer.
- Stateless: no conversation history and no prior result context.
- Exactly one generation per request; no voting, retries, or self-correction.
- Strict JSON-schema response format.
- No tool execution capability.

## Configuration differences that could not be eliminated

| Setting | Qwen3.8-27B | GPT-OSS-20B |
|---|---|---|
| OpenRouter model | `qwen/qwen3.8-27b` | `openai/gpt-oss-20b` |
| Reasoning | Disabled | Mandatory low effort, hidden from output |
| Completion ceiling | 160 | 256 |
| Provider routing | Cheapest compatible, multiple providers | DeepInfra pinned, no fallback |
| Providers observed | Chutes, CoreWeave, Reka | DeepInfra |

GPT-OSS rejected requests when reasoning was set to `none`. The smallest supported setting, `low`, was used with reasoning excluded from visible output. The 256-token ceiling was necessary because hidden reasoning consumes the completion budget. Consequently, generated-token counts are not a pure apples-to-apples comparison.

DeepInfra was pinned after the cheapest compatible provider, Darkbloom, returned valid final JSON for only 3/18 smoke cases. The failed provider smoke is preserved separately and is not included in the model comparison.

## Overall Stage A results

| Metric | Qwen3.8-27B | GPT-OSS-20B | GPT minus Qwen |
|---|---:|---:|---:|
| Valid delegation JSON | 100.0% (450/450) | 99.8% (449/450) | -0.2 pp |
| Malformed delegation | 0.0% | 0.2% | +0.2 pp |
| Correct domain | 99.1% (446/450) | 86.2% (388/450) | **-12.9 pp** |
| Semantic-request heuristic | 69.3% | 47.3% | -22.0 pp |
| Average verbatim recall | 90.6% | 83.4% | -7.1 pp |
| Average verbatim precision | 75.9% | 55.3% | -20.6 pp |
| Final-tool leakage | 0.2% | 0.0% | -0.2 pp |
| Average latency | 2.145 s | 1.032 s | -1.113 s |
| p50 latency | 1.210 s | 0.918 s | -0.291 s |
| p95 latency | 6.976 s | 1.885 s | -5.091 s |
| Average generated tokens | 28.96 | 74.45 | not comparable directly |
| Reported generation speed | 21.25 tok/s | 74.48 tok/s | 3.51x |
| API cost for 450 | $0.13307 | $0.01207 | -90.9% |

The deterministic semantic-request score is a lexical-cue heuristic, not an LLM judge. It can reject useful concise wording such as `read file` simply because it lacks one of the scorer's registered phrases. Domain accuracy and literal payload measurements are the more dependable Stage A comparisons.

## Paired case comparison

### Domain decisions

| Paired outcome | Cases |
|---|---:|
| Both correct | 388 |
| Qwen correct, GPT-OSS wrong | 58 |
| GPT-OSS correct, Qwen wrong | 0 |
| Both wrong | 4 |

GPT-OSS did not recover any of Qwen's four domain failures. It repeated all four and added 58 more.

The four cases both models missed were:

1. `p2-search-12`: “dig up recent notes” was interpreted as memory/Yuki-file retrieval rather than web information search.
2. `p2-see-23`: “Locate AirPods Pro” lacked an explicit camera cue and was interpreted as information/Yuki-file search.
3. `p2-yuki_write-22`: a bare exact `filename|content` preservation request was interpreted as memory storage.
4. `p2-yuki_append-23`: a bare exact `filename|payload` preservation request was interpreted as memory storage.

### Exact verbatim arrays

| Paired outcome | Cases |
|---|---:|
| Both exact | 225 |
| Qwen exact, GPT-OSS not exact | 111 |
| GPT-OSS exact, Qwen not exact | 9 |
| Neither exact | 105 |

GPT-OSS did improve nine individual exact payloads, including several search strings, `hostname`, one image prompt, one append payload, and one recall topic. Those local wins were outweighed by 111 cases where Qwen was exact and GPT-OSS was not.

## Accuracy by schema domain

| Expected domain | Cases | Qwen domain | GPT-OSS domain | Delta |
|---|---:|---:|---:|---:|
| Information | 125 | 99.2% | 92.0% | -7.2 pp |
| Computer | 75 | 100.0% | 96.0% | -4.0 pp |
| Media | 50 | 98.0% | 72.0% | **-26.0 pp** |
| Yuki files | 125 | 98.4% | 72.0% | **-26.4 pp** |
| Memory | 50 | 100.0% | 100.0% | 0.0 pp |
| External agent | 25 | 100.0% | 100.0% | 0.0 pp |

The overall loss is not uniform. GPT-OSS is strong at identifying memory and external-agent domains, but weak at Yuki's custom boundaries:

- Yuki-owned files versus the normal computer/filesystem.
- Existing visual inspection versus delegating to an external agent.
- Image generation versus external-agent work.

## Domain confusion matrix

### Qwen3.8-27B

| Expected | Correct | Wrong destinations |
|---|---:|---|
| Information | 124/125 | 1 memory |
| Computer | 75/75 | none |
| Media | 49/50 | 1 information |
| Yuki files | 123/125 | 2 memory |
| Memory | 50/50 | none |
| External agent | 25/25 | none |

### GPT-OSS-20B

| Expected | Correct | Wrong destinations |
|---|---:|---|
| Information | 115/125 | 6 external agent, 2 computer, 1 memory, 1 Yuki files |
| Computer | 72/75 | 2 Yuki files, 1 external agent |
| Media | 36/50 | 11 external agent, 2 Yuki files, 1 computer |
| Yuki files | 90/125 | 29 computer, 4 memory, 1 external agent, 1 malformed |
| Memory | 50/50 | none |
| External agent | 25/25 | none |

The largest single confusion is Yuki files → computer: 29 cases. The next is media → external agent: 11 cases.

## Per-tool domain and semantic results

Each row contains 25 frozen cases.

| Tool | Qwen domain | GPT domain | Delta | Qwen semantic heuristic | GPT semantic heuristic |
|---|---:|---:|---:|---:|---:|
| `time` | 100% | 96% | -4 pp | 80% | 76% |
| `weather` | 100% | 96% | -4 pp | 100% | 96% |
| `fetch` | 100% | 92% | -8 pp | 92% | 80% |
| `search` | 96% | 80% | -16 pp | 8% | 4% |
| `calc` | 100% | 96% | -4 pp | 100% | 88% |
| `hardware` | 100% | 92% | -8 pp | 84% | 76% |
| `see` | 96% | 64% | **-32 pp** | 80% | 24% |
| `image` | 100% | 80% | -20 pp | 96% | 40% |
| `read` | 100% | 96% | -4 pp | 80% | 12% |
| `shell` | 100% | 100% | 0 pp | 96% | 92% |
| `yuki_write` | 96% | 96% | 0 pp | 24% | 0% |
| `yuki_read` | 100% | 64% | **-36 pp** | 0% | 0% |
| `yuki_list` | 100% | 44% | **-56 pp** | 36% | 24% |
| `yuki_delete` | 100% | 64% | **-36 pp** | 0% | 0% |
| `yuki_append` | 96% | 92% | -4 pp | 88% | 20% |
| `remember` | 100% | 100% | 0 pp | 100% | 100% |
| `recall` | 100% | 100% | 0 pp | 88% | 24% |
| `ask_claude` | 100% | 100% | 0 pp | 96% | 96% |

The semantic heuristic severely under-scores several valid concise Yuki-file delegations for both models. However, the large domain losses for `see`, `yuki_read`, `yuki_list`, and `yuki_delete` are exact label failures and are not artifacts of that heuristic.

## Exact-payload transport

The dataset marks 300/450 cases as requiring one or more exact strings in `verbatim`. The other 150 cases have no required exact span under the frozen annotations.

| Payload metric | Qwen | GPT-OSS |
|---|---:|---:|
| Payload-bearing cases | 300 | 300 |
| Complete required-span recall | 256/300 (85.3%) | 220/300 (73.3%) |
| Exact expected verbatim array | 251/300 (83.7%) | 211/300 (70.3%) |
| Required-payload omission cases | 44 | 80 |
| Exact array across all 450 cases | 336/450 (74.7%) | 234/450 (52.0%) |

The aggregate precision metric also penalizes extra strings in cases where the frozen dataset did not require a verbatim payload. For example, copying a weather city can score as unnecessary even though it may help the dispatcher. Exact required-span recall and the payload-bearing exact-array result are therefore the cleaner preservation measures.

### Exact array by payload-bearing tool

| Tool | Required cases | Qwen exact | GPT exact | Delta |
|---|---:|---:|---:|---:|
| `fetch` | 25 | 25 (100%) | 25 (100%) | 0 |
| `search` | 25 | 15 (60%) | 17 (68%) | +2 |
| `image` | 25 | 15 (60%) | 8 (32%) | -7 |
| `read` | 25 | 25 (100%) | 25 (100%) | 0 |
| `shell` | 25 | 22 (88%) | 23 (92%) | +1 |
| `yuki_write` | 25 | 21 (84%) | 18 (72%) | -3 |
| `yuki_read` | 25 | 25 (100%) | 23 (92%) | -2 |
| `yuki_delete` | 25 | 24 (96%) | 23 (92%) | -1 |
| `yuki_append` | 25 | 19 (76%) | 10 (40%) | -9 |
| `remember` | 25 | 24 (96%) | 18 (72%) | -6 |
| `recall` | 25 | 20 (80%) | 7 (28%) | **-13** |
| `ask_claude` | 25 | 16 (64%) | 14 (56%) | -2 |

GPT-OSS slightly outperformed Qwen on exact `search` and `shell` spans. Its major payload regressions were `recall`, `yuki_append`, `image`, and `remember`.

Typical GPT-OSS behavior was over-capturing the surrounding sentence instead of extracting the minimal payload:

```text
User: Generate a camera-shaped sculpture; don't use the real camera.

Gold verbatim:
["camera-shaped sculpture"]

GPT-OSS:
["Generate a camera-shaped sculpture; don't use the real camera."]
```

For recall, it commonly included conversational framing:

```text
User: Can you dig into our shared past for the database migration topic?

Gold verbatim:
["database migration"]

GPT-OSS:
["our shared past for the database migration topic"]
```

Deterministic source binding can potentially repair minimal-span extraction after tool selection, but it cannot repair a wrong domain that prevents the correct schema from reaching Hammer.

## Robustness by dataset category

| Category | Cases | Qwen domain | GPT domain | Qwen verbatim recall | GPT verbatim recall |
|---|---:|---:|---:|---:|---:|
| Normal | 202 | 100.0% | 87.6% | 89.4% | 82.7% |
| Paraphrase | 110 | 99.1% | 86.4% | 93.6% | 85.9% |
| Confusion | 72 | 100.0% | 87.5% | 93.1% | 80.6% |
| Preservation | 48 | 93.8% | 72.9% | 84.4% | 84.4% |
| Adversarial | 18 | 100.0% | 100.0% | 91.7% | 86.1% |

GPT-OSS did not fail primarily on the explicitly adversarial cases. Its weakness appears throughout normal, paraphrase, and confusion language, with the lowest domain score on preservation cases.

## Representative semantic failures

### `see` interpreted as external-agent work

```text
User: Use your eyes and track down my earbuds.

Qwen:
{"request":"Use camera/vision to locate the user's earbuds","domain_hint":"media","verbatim":["earbuds"]}

GPT-OSS:
{"request":"search for user’s earbuds location","domain_hint":"external_agent","verbatim":["Use your eyes and track down my earbuds."]}
```

### Yuki file interpreted as ordinary filesystem work

```text
User: Pull up your own little file called snack.txt.

Qwen domain: yuki_files
GPT-OSS domain: computer
```

### `yuki_list` completion-budget failure

`p2-yuki_list-06` consumed the 256-token completion budget and returned a truncated JSON object. The provider reported 280 reasoning tokens for that generation.

## Existing Qwen → Hammer 1.5B downstream baseline

This is included for architectural context. It is not a direct GPT comparison because GPT-OSS delegations have not yet been run through Hammer.

| Qwen → Hammer metric | Frozen full-450 result |
|---|---:|
| Qwen domain accuracy | 99.1% |
| Hammer final-tool selection | 90.7% |
| Strict argument accuracy | 79.6% |
| Preservation-aware argument accuracy | 80.9% |
| Strict exact call | 79.6% (358/450) |
| Preservation-aware exact call | 80.9% (364/450) |
| Schema-valid output | 92.4% |
| Malformed Hammer output | 0.4% |
| Hammer average latency | 2.486 s |
| Hammer peak MLX memory | 3.473 GB |

Because GPT-OSS Stage A exposes the correct domain in only 388 cases, its current theoretical final-tool ceiling is 388/450. That is already 20 cases below Qwen → Hammer's measured 408/450 correct tools, before accounting for any Hammer mistakes.

## Efficiency and cost

GPT-OSS's efficiency advantage is real:

- Average delegation latency was 51.9% lower.
- p95 latency was 73.0% lower.
- Reported token throughput was 3.51x higher.
- API cost was 90.9% lower, approximately 11.0x cheaper.

However, the comparison is influenced by provider policy and mandatory reasoning. Qwen was routed across three providers, while GPT-OSS was pinned to DeepInfra after Darkbloom failed the structured-output smoke. Treat the latency and cost figures as observed deployment results, not provider-independent model constants.

## What the result means for Yuki

### GPT-OSS strengths

- Very low API cost.
- Faster and more stable tail latency in this provider configuration.
- Almost-perfect structured JSON output after provider pinning.
- Perfect broad-domain recognition for memory and external-agent requests.
- Strong exact URL, normal filepath, and shell-command transport.
- Zero final-tool-name leakage in the semantic request.

### GPT-OSS weaknesses

- Large Yuki-files versus normal-computer confusion.
- Large `see`/media versus external-agent confusion.
- Worse image, recall, remember, and append payload extraction.
- Tendency to copy entire clauses or sentences instead of minimal exact spans.
- Mandatory hidden reasoning complicates latency/token controls and caused one truncation even at 256 tokens.

## Decision

Do not replace Qwen3.8-27B with GPT-OSS-20B as Yuki's semantic main brain under the current contract.

The result is strong enough that a GPT-OSS → Hammer full run is not required to reach that main-brain decision: its Stage A domain ceiling is already lower than Qwen → Hammer's measured final tool-selection accuracy.

A useful next GPT-OSS experiment would be architecturally different:

```text
raw user request
  -> GPT-OSS with all 18 real Yuki tool schemas
  -> native structured tool call
  -> parse + validate + score
```

That would test whether GPT-OSS's native tool training performs better when it selects the final function directly, without the six-domain semantic-delegation interface. It should be treated as a new dispatcher experiment, not as evidence against or for the result in this report.

## Artifacts and checksums

- GPT-OSS full raw report: `stage-a/mainbrain-gpt-oss-20b-openrouter-full450-v1-deepinfra.json`
  - SHA-256: `f99588031a30cf8fbdbed966c07366fc7fcdeb9e793acadcbdc4c423e2a42cf7`
- Qwen full raw report: `../production-delegation/stage-a/production-delegation-mainbrain-qwen3.8-27b-openrouter-full450-v1.json`
  - SHA-256: `ee68fa403cc3a6d211aa0ffad99a474794378e5bb94f0cbb70b8f94c363be5fa`
- Qwen → Hammer raw report: `../production-delegation/stage-b/production-delegation-hammer1.5b-qwen3.8-27b-full450-v1.json`
  - SHA-256: `84b619922e3b24c0080b17b91a5963114f024647112e467a6099ae9fab425f8a`
- Machine-readable GPT/Qwen comparison: `GPT-OSS-20B-VS-QWEN38-STAGE-A-COMPARISON-v1.json`

No frozen input or previous report was modified.
