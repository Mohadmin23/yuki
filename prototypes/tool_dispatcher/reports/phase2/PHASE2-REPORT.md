# Yuki Tool Dispatcher — Phase 2 Reliability Report

Date: 2026-08-19  
Dataset: 450 frozen, hand-authored routing cases  
Dataset SHA-256: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`

## Executive result

Phase 2 does **not** establish a production-ready winner because the unchanged automatic schema selector failed to offer the expected tool in 129 of 450 cases (71.33% coverage). That selector failure limits end-to-end strict exact-call accuracy to 60.00% for Hammer 3B and 60.89% for Arch-Agent 3B.

The finalist comparison itself is mixed:

- **Arch-Agent 3B selects tools better:** 70.67% end to end and 99.07% when the expected tool was available, versus Hammer's 66.67% and 93.46%.
- **Arch-Agent 3B has better execution-equivalent complete calls:** 63.33% versus 60.44%.
- **Strict complete-call reliability is effectively tied:** Arch 60.89% versus Hammer 60.00%; the paired difference is not significant on this dataset (`p = 0.6655`).
- **Hammer 3B preserves exact arguments better after selecting the right tool:** 90.00% versus 86.16%.
- **Hammer 3B obeys the structured-only contract better:** zero malformed or conversational generations, versus 61 malformed Arch generations, 60 of which were conversational. Most Arch formatting failures happened after the selector had already omitted the expected tool, but they remain contract violations.
- **Hammer 3B is faster:** 4.466 s average total routing latency versus 5.372 s for Arch, with higher generation throughput.

Technical recommendation: keep **Hammer 3B as the default isolated-prototype candidate**, not as a production integration. It is the safer default under the dispatcher's hard structured-only contract and is faster, while giving up only 4 strict successes across 450 cases. Arch-Agent 3B is the stronger tool chooser and should remain the comparison candidate when the selector is redesigned and held-out validation is rerun.

No Yuki integration, model tuning, Phase 3 work, or xLAM run was performed.

## Experimental contract and safety

Each case followed exactly this path:

```text
request
  -> unchanged deterministic automatic schema selector
  -> selected live Yuki schema subset
  -> one stateless native dispatcher generation
  -> native parser and canonical normalization
  -> strict schema validation
  -> deterministic benchmark scoring
  -> stop
```

The run used the live 18-tool Yuki registry and prototype validation adapter. Each model used its already verified FP16 MLX checkpoint, native output dialect, existing prompt/adapter, and a 256-token generation ceiling. Cases were independent: no conversation history, previous requests, tool results, answer generation, or output repair.

The Phase 2 runner exposes no execution argument or execution capability. Tests also replace the underlying execution function with a hard failure and prove the benchmark still completes. No Yuki tool was invoked: no tool-level shell command, user-file operation, camera, memory operation, API, web service, Claude bridge, image generator, or other external service call occurred.

Hammer was run through all 450 cases first from `/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-3b-fp16`. Its process was then gone before `/Volumes/madisk/yuki-tool-dispatcher/Arch-Agent-3B-fp16` was loaded and run. The two model reports contain the same case identities, expected calls, selected domains, offered schemas, scoring rules, and dataset checksum. Both checkpoints remained on the external disk.

## Dataset

The case file was finalized and checksummed before either finalist was run. It contains 25 cases for each of the 18 live tools, with no duplicate request strings.

| Category | Cases | Share |
|---|---:|---:|
| Normal/common | 202 | 44.89% |
| Natural paraphrase | 110 | 24.44% |
| Known confusion boundary | 72 | 16.00% |
| Argument preservation | 48 | 10.67% |
| Adversarial/ambiguous/incomplete | 18 | 4.00% |
| **Total** | **450** | **100.00%** |

The requests vary wording, syntax, directness, capitalization, punctuation, typos, path and filename forms, URLs, quoted content, symbols, long payloads, and distracting text. This is a balanced reliability set, not a sample of Yuki's real deployment frequency. Its confidence intervals describe uncertainty for these cases; they should not be treated as population guarantees.

## Automatic selector result

The unchanged selector did **not** generalize from its 168/168 development result.

| Selector metric | Phase 2 result |
|---|---:|
| Expected tool offered | 321/450 (71.33%) |
| Exact expected domain | 291/450 (64.67%) |
| Average selected domains | 1.069 |
| Average schemas offered | 4.347 |
| Schema range | 1–8 |
| Average selector latency | 0.057–0.060 ms |
| p50 selector latency | 0.056–0.057 ms |
| p95 selector latency | 0.096–0.098 ms |

Selector coverage by expected tool:

| Tool | Offered | Tool | Offered | Tool | Offered |
|---|---:|---|---:|---|---:|
| time | 23/25 | weather | 24/25 | fetch | 23/25 |
| search | 23/25 | calc | 22/25 | hardware | 12/25 |
| see | 17/25 | image | 5/25 | read | 25/25 |
| shell | 22/25 | yuki_write | 19/25 | yuki_read | 19/25 |
| yuki_list | 13/25 | yuki_delete | 17/25 | yuki_append | 17/25 |
| remember | 10/25 | recall | 11/25 | ask_claude | 19/25 |

Coverage was especially weak for paraphrases (53.64%), media, memory, hardware, and Yuki file listing. Because both finalists received the selector's identical subsets, this does not bias their paired comparison, but it makes the current combined architecture unreliable.

## Core reliability

All accuracy figures below are end to end over 450 cases unless marked conditional. The intervals are 95% Wilson intervals.

| Metric | Hammer 3B | Arch-Agent 3B | Difference (Arch − Hammer) |
|---|---:|---:|---:|
| Expected tool offered | 71.33% | 71.33% | 0.00 pp |
| Tool selection | 66.67% (62.19–70.86%) | **70.67%** (66.30–74.68%) | +4.00 pp |
| Tool selection, expected tool available | 93.46% | **99.07%** | +5.61 pp |
| Strict argument accuracy | 60.00% | **60.89%** | +0.89 pp |
| Equivalent argument accuracy | 60.44% | **63.33%** | +2.89 pp |
| Strict arguments, correct tool selected | **90.00%** | 86.16% | −3.84 pp |
| Equivalent arguments, correct tool selected | **90.67%** | 89.62% | −1.05 pp |
| Strict exact call | 60.00% (55.41–64.42%) | **60.89%** (56.31–65.29%) | +0.89 pp |
| Equivalent exact call | 60.44% (55.86–64.86%) | **63.33%** (58.79–67.66%) | +2.89 pp |
| Strict exact, expected tool available | 84.11% | **85.36%** | +1.25 pp |
| Equivalent exact, expected tool available | 84.74% | **88.79%** | +4.05 pp |
| Schema-valid output | 67.56% | **85.33%** | +17.78 pp |
| Malformed output | **0.00%** | 13.56% | +13.56 pp |
| Conversational output | **0.00%** | 13.33% | +13.33 pp |
| Nonexistent tool | **0.00%** | 1.11% | +1.11 pp |
| Multiple calls | 0.00% | 0.00% | 0.00 pp |

The schema-valid rate needs context. When Hammer was offered an irrelevant subset, it normally emitted its native empty-array rejection, which parsed cleanly but was intentionally not a valid executable call. Arch often called an irrelevant offered tool, producing a schema-valid but incorrect call, or replied conversationally. Schema validity therefore must not be read as task correctness.

Of Arch's 61 malformed outputs, 58 occurred when the selector had already omitted the expected tool. The three expected-tool-available formatting failures were one `fetch` refusal, one `read` refusal, and one malformed `ask_claude` call containing unescaped inner JSON quotes.

## Paired comparison

| Dimension | Both correct | Hammer only | Arch only | Both wrong, same | Both wrong, different | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Strict exact call | 248 | 22 | 26 | 9 | 145 | 0.6655 |
| Equivalent exact call | 260 | 12 | 25 | 9 | 144 | 0.0470 |
| Tool selection | 298 | 2 | 20 | 63 | 67 | 0.000121 |

Arch's tool-selection advantage is clear on this fixed case set. Its execution-equivalent advantage is marginal at the conventional 0.05 threshold. The four-case net strict advantage is not statistically distinguishable. These tests do not turn the hand-designed dataset into a random sample of future requests.

## Performance on the M1 Air

| Metric | Hammer 3B | Arch-Agent 3B |
|---|---:|---:|
| Dispatcher average | **4,464.8 ms** | 5,370.8 ms |
| Dispatcher p50 | **4,482.6 ms** | 5,089.9 ms |
| Dispatcher p95 | **6,817.2 ms** | 8,138.6 ms |
| Total routing average | **4,465.6 ms** | 5,371.8 ms |
| Total routing p50 | **4,483.4 ms** | 5,090.9 ms |
| Total routing p95 | **6,817.9 ms** | 8,139.8 ms |
| Average generation throughput | **10.54 tok/s** | 9.29 tok/s |
| Median generation throughput | **9.83 tok/s** | 9.27 tok/s |
| Average prompt throughput | **258.45 tok/s** | 219.49 tok/s |
| Average prompt tokens | 648.3 | 648.6 |
| Peak MLX memory | 6.533 GB | 6.541 GB |
| Maximum sampled process RSS | 1,134 MB | 602 MB |

Hammer's average total latency was 906 ms lower (16.87% lower than Arch), and its p95 was 1.322 s lower. Both fit in the machine's memory envelope. MLX's allocated-memory metric is the useful accelerator comparison; process RSS sampling does not include unified-memory accounting in the same way and should not be used to claim Arch uses half the real memory.

## Per-tool results

Percentages are end to end and include selector omissions. `Valid` means the generated call passed the offered tool's schema; it does not imply that the expected tool was selected. Latency is average total routing latency.

### Hammer 3B

| Tool | N | Tool selected | Strict exact | Equivalent exact | Valid | Malformed | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| time | 25 | 84% | 84% | 84% | 84% | 0% | 3,634 ms |
| weather | 25 | 88% | 88% | 88% | 88% | 0% | 4,262 ms |
| fetch | 25 | 92% | 92% | 92% | 92% | 0% | 5,135 ms |
| search | 25 | 72% | 56% | 60% | 72% | 0% | 4,161 ms |
| calc | 25 | 88% | 80% | 84% | 88% | 0% | 4,869 ms |
| hardware | 25 | 48% | 48% | 48% | 48% | 0% | 3,340 ms |
| see | 25 | 56% | 56% | 56% | 56% | 0% | 3,301 ms |
| image | 25 | 16% | 4% | 4% | 16% | 0% | 3,138 ms |
| read | 25 | 84% | 84% | 84% | 84% | 0% | 4,543 ms |
| shell | 25 | 84% | 80% | 80% | 84% | 0% | 4,625 ms |
| yuki_write | 25 | 76% | 64% | 64% | 76% | 0% | 5,739 ms |
| yuki_read | 25 | 64% | 64% | 64% | 68% | 0% | 4,828 ms |
| yuki_list | 25 | 52% | 52% | 52% | 56% | 0% | 4,086 ms |
| yuki_delete | 25 | 68% | 64% | 64% | 68% | 0% | 4,877 ms |
| yuki_append | 25 | 68% | 48% | 48% | 68% | 0% | 5,974 ms |
| remember | 25 | 40% | 36% | 36% | 40% | 0% | 4,511 ms |
| recall | 25 | 44% | 40% | 40% | 52% | 0% | 4,569 ms |
| ask_claude | 25 | 76% | 40% | 40% | 76% | 0% | 4,789 ms |

### Arch-Agent 3B

| Tool | N | Tool selected | Strict exact | Equivalent exact | Valid | Malformed | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| time | 25 | 92% | 92% | 92% | 100% | 0% | 3,937 ms |
| weather | 25 | 96% | 96% | 96% | 100% | 0% | 5,025 ms |
| fetch | 25 | 88% | 88% | 88% | 92% | 8% | 5,847 ms |
| search | 25 | 92% | 60% | 60% | 100% | 0% | 5,418 ms |
| calc | 25 | 88% | 60% | 88% | 100% | 0% | 5,519 ms |
| hardware | 25 | 48% | 48% | 48% | 68% | 32% | 5,406 ms |
| see | 25 | 68% | 60% | 60% | 80% | 20% | 4,661 ms |
| image | 25 | 20% | 8% | 8% | 24% | 72% | 8,043 ms |
| read | 25 | 96% | 96% | 96% | 96% | 4% | 4,439 ms |
| shell | 25 | 88% | 84% | 84% | 92% | 8% | 4,732 ms |
| yuki_write | 25 | 76% | 56% | 56% | 88% | 12% | 6,348 ms |
| yuki_read | 25 | 76% | 76% | 76% | 88% | 8% | 5,493 ms |
| yuki_list | 25 | 52% | 52% | 52% | 68% | 32% | 5,500 ms |
| yuki_delete | 25 | 68% | 68% | 68% | 68% | 32% | 5,884 ms |
| yuki_append | 25 | 68% | 56% | 56% | 96% | 4% | 6,308 ms |
| remember | 25 | 40% | 40% | 40% | 80% | 8% | 4,845 ms |
| recall | 25 | 44% | 32% | 32% | 100% | 0% | 4,874 ms |
| ask_claude | 25 | 72% | 24% | 40% | 96% | 4% | 4,416 ms |

## Tool-family reliability

| Family | N | Hammer tool / strict / equivalent | Arch tool / strict / equivalent | Technical read |
|---|---:|---:|---:|---|
| Information | 125 | 84.8% / **80.0%** / 81.6% | **91.2%** / 79.2% / **84.8%** | Arch routes better; strict tie-like |
| Computer | 75 | 72.0% / 70.7% / 70.7% | **77.3% / 76.0% / 76.0%** | Arch |
| Media | 50 | 36.0% / 30.0% / 30.0% | **44.0% / 34.0% / 34.0%** | Arch, but selector dominates |
| Yuki files | 125 | 65.6% / 58.4% / 58.4% | **68.0% / 61.6% / 61.6%** | Arch narrowly |
| Memory | 50 | **42.0% / 38.0% / 38.0%** | 42.0% / 36.0% / 36.0% | Tool tie; Hammer calls |
| External agent | 25 | **76.0% / 40.0% / 40.0%** | 72.0% / 24.0% / **40.0%** | Hammer strict; equivalent tie |

These family figures include selector failure and small subgroup sizes. They identify where to investigate; they are not separate model leaderboards.

## Failure taxonomy and confusion analysis

Every strict failure has one primary category and may have secondary tags.

| Primary failure | Hammer | Arch-Agent |
|---|---:|---:|
| Selector omitted expected tool | 129 | 129 |
| Wrong tool | 21 | 0 primary / 3 secondary |
| Wrong argument | 28 | 33 |
| Semantically equivalent argument rewrite | 2 | 11 |
| Conversational output | 0 | 2 primary / 60 total tags |
| Malformed output | 0 | 1 primary / 61 total tags |
| Nonexistent tool | 0 | 0 primary / 5 secondary |
| Missing argument | 0 | 0 |
| Extra argument | 0 | 0 |
| Other schema invalid | 0 | 0 |
| Multiple calls | 0 | 0 |
| **Strict failures** | **180** | **176** |

Primary precedence attributes a case to the selector first. This is why Arch's many conversational/malformed secondary tags do not inflate its primary counts when the expected tool was already unavailable.

Largest remaining failures:

1. **Selector recall/generalization:** 129 omissions. `image` (20), `remember` (15), `recall` (14), `hardware` (13), and `yuki_list` (12) were the largest tool-level gaps. Paraphrase coverage was only 53.64%.
2. **Hammer model selection after availability:** 21 of 321 available cases chose or returned the wrong tool/reject outcome. Arch missed only 3.
3. **Argument preservation:** among correct tool selections, Hammer had 30 strict argument misses and Arch had 44. Hammer's failures concentrated in `ask_claude`, Yuki append/write payloads, search, and image prompts. Arch's concentrated in search queries, `ask_claude`, calc normalization, Yuki payloads, and image/vision prompts.
4. **Arch structured-output discipline:** it produced 60 conversational responses and five nonexistent placeholder tool names. Fifty-eight malformed cases followed selector omission, but three occurred when the expected tool was available.
5. **Media and memory:** both families are poor end to end primarily because the selector often excludes their tools.

Focused end-to-end confusion/error counts, including selector-driven `<none>` outcomes:

| Cluster | Hammer errors | Arch errors | Main observation |
|---|---:|---:|---|
| `read` / `yuki_read` / `shell` | 17 | 10 | Arch recovers more offered-tool cases; most remaining errors follow omissions |
| `weather` / `search` / `fetch` | 12 | 6 | Arch routes information requests better |
| `see` / `image` | 32 | 28 | Selector omission dominates, especially `image` |
| `remember` / `recall` | 29 | 29 | Same error count; Arch more often calls another offered tool, Hammer rejects |
| `yuki_write` / `yuki_append` | 14 | 14 | Same tool-level error count; argument accuracy separates calls |

The full 18-by-output matrices and exact error destinations are in `phase2-tool-confusion.json`.

## Answers to the 12 Phase 2 questions

1. **Which model has higher strict exact-call reliability?** Arch-Agent, 60.89% versus 60.00%, but only by 4/450 cases. The paired result is not significant (`p = 0.6655`), so this is an observed numerical lead, not evidence of a dependable strict advantage.

2. **Which has higher execution-equivalent reliability?** Arch-Agent, 63.33% versus 60.44%. It won 25 discordant cases to Hammer's 12 (`p = 0.0470`), a modest and borderline result on this hand-designed set.

3. **Which has better tool-selection reliability?** Arch-Agent: 70.67% versus 66.67% end to end, and 99.07% versus 93.46% when the expected tool was offered. Paired tool-selection outcomes were 20 Arch-only versus 2 Hammer-only (`p = 0.000121`).

4. **Which preserves exact arguments better?** Hammer. Conditional on selecting the expected tool, strict argument accuracy was 90.00% versus Arch's 86.16%. Hammer also led slightly under equivalence, 90.67% versus 89.62%.

5. **Which produces fewer malformed/conversational outputs?** Hammer, decisively: zero malformed and zero conversational outputs. Arch produced 61 malformed and 60 conversational outputs. Most were triggered by selector omissions, but Hammer handled the same omissions with structured rejection.

6. **Which is faster?** Hammer. Average routing was 4.466 s versus 5.372 s, p95 was 6.818 s versus 8.140 s, and generation throughput was 10.54 versus 9.29 tok/s.

7. **Which is more reliable by tool family?** Arch leads equivalent exact accuracy for information, computer, media, and Yuki files. Hammer leads memory slightly and strict `ask_claude` handling; external-agent equivalent accuracy ties. Family conclusions remain selector-limited.

8. **What are the largest remaining failure modes?** Selector omission first; then Hammer's available-tool selection/rejection errors, argument preservation for both models, and Arch's conversational/malformed behavior when it lacks the right schema.

9. **Does the automatic selector remain reliable at Phase 2 scale?** No. Coverage fell from 168/168 in the earlier development evaluation to 321/450 (71.33%) on the frozen Phase 2 set. It requires redesign and genuinely held-out validation before this architecture can be trusted.

10. **Are the differences large enough to justify one model over the other?** Not on strict complete-call accuracy. Arch's tool-selection advantage is strong, and its equivalent-call advantage is modest; Hammer's structured-only compliance, exact argument preservation, and latency advantages are also material. The choice depends on which dispatcher contract is prioritized, and selector failure currently overwhelms both.

11. **Should xLAM-2 3B return as a third finalist?** Not yet. A third dispatcher cannot repair the selector's 28.67% omission rate, and it would not resolve the mixed contract-versus-routing tradeoff cleanly. Keep xLAM available, repair and freeze the selector on separate development data, then decide whether a three-way held-out rerun is worth the heat and time.

12. **Which should be the default Yuki dispatcher based only on measured technical evidence?** **Hammer 3B should remain the default prototype candidate**, because the architecture's hard requirement is structured tool-call-only behavior, Hammer had zero violations, it preserved exact arguments better, and it was faster. This is not authorization to integrate it. If a future selector is reliable and a hard structural constraint/guard is added, Arch's significantly better tool selection makes it a serious candidate to overtake Hammer.

## Licensing and deployment note

Licensing does not change the technical ranking above, but neither checkpoint should be assumed commercially deployable without review:

- [MadeAgents/Hammer2.1-3b](https://huggingface.co/MadeAgents/Hammer2.1-3b) identifies its license as the Qwen Research License. The linked terms limit the supplied model materials to non-commercial research/evaluation and direct commercial users to request a separate license.
- [katanemo/Arch-Agent-3B](https://huggingface.co/katanemo/Arch-Agent-3B) identifies its license as the Katanemo Community License. Its current linked terms require a separate DigitalOcean commercial license and contain attribution/redistribution conditions.

This local research benchmark fits the stated evaluation purpose. Any distribution or commercial use needs a fresh license review against the exact checkpoint and then-current terms.

## Limitations and stopping point

- The dataset is hand-designed and balanced at 25 cases per tool; it does not estimate Yuki's real request distribution.
- The previous 168-case selector result was development evidence, not a reliable held-out estimate. Phase 2 exposed substantial selector overfitting or coverage gaps.
- No tool execution was measured, by design. External-service reliability and Yuki integration are outside this result.
- No prompt, selector, adapter, scoring, generation, model, or dataset changes were made between finalists.
- xLAM was not run. No model was fine-tuned, quantized, or integrated into Yuki.
- The experiment stops here pending a new instruction.
