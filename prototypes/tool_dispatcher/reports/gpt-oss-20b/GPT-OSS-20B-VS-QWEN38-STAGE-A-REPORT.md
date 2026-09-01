# GPT-OSS-20B vs Qwen3.8-27B — Yuki Stage A

## Outcome

GPT-OSS-20B is much faster and cheaper than Qwen3.8-27B in Yuki's frozen semantic-delegation stage, but it is materially less reliable. Under the unchanged 450-case contract, domain accuracy fell from **99.1% to 86.2%**.

This test replaces Qwen only in Stage A:

```text
raw user request
  -> semantic main-brain delegation
  -> parse and score
```

No Yuki tool was executed. Hammer was not run because the external volume containing the Hammer 1.5B checkpoint was not mounted.

## Controlled setup

- Frozen dataset: 450 cases, 25 per Yuki tool.
- Same source checksum: `ea6711ce0548d3771166f8f2f9a5e59ebd73acb14a9ddf7ecbf2f6589f1744c3`.
- Same system prompt checksum: `1189b8cb529f8f13592e4952b9e1da31093592dfe2710c96796bc9bedb120cbb`.
- Same strict JSON schema, parser, and scorer.
- Stateless, one generation per case, no retries or voting.
- GPT-OSS provider pinned to DeepInfra with fallback disabled.
- GPT-OSS requires reasoning, so low reasoning was enabled and excluded from visible output. It used a 256-token ceiling. Qwen had reasoning disabled and used the frozen 160-token configuration. Generated-token counts are therefore not directly comparable.

## Headline comparison

| Metric | Qwen3.8-27B | GPT-OSS-20B |
|---|---:|---:|
| Valid delegation | 100.0% | 99.8% |
| Domain accuracy | 99.1% | 86.2% |
| Semantic-request quality | 69.3% | 47.3% |
| Average verbatim recall | 90.6% | 83.4% |
| Average verbatim precision | 75.9% | 55.3% |
| Final-tool leakage | 0.2% | 0.0% |
| Average latency | 2.145 s | 1.032 s |
| p50 latency | 1.210 s | 0.918 s |
| p95 latency | 6.976 s | 1.885 s |
| Generation throughput | 21.25 tok/s | 74.48 tok/s |
| API cost, 450 cases | $0.1331 | $0.0121 |

GPT-OSS used roughly **48% of Qwen's average latency**, delivered **3.5x** the reported generation throughput, and cost about **9.1%** as much. Those efficiency gains did not compensate for the 12.9-point domain-accuracy loss.

## Where GPT-OSS failed

The 62 incorrect domain decisions were concentrated in Yuki-specific ontology boundaries:

| Tool | Correct domain | Failures | Accuracy |
|---|---:|---:|---:|
| `yuki_list` | yuki_files | 14/25 | 44% |
| `see` | media | 9/25 | 64% |
| `yuki_read` | yuki_files | 9/25 | 64% |
| `yuki_delete` | yuki_files | 9/25 | 64% |
| `search` | information | 5/25 | 80% |
| `image` | media | 5/25 | 80% |

The dominant confusions were:

- Yuki-local file actions interpreted as ordinary `computer` filesystem work.
- Visual inspection (`see`) interpreted as `external_agent` delegation.
- Image generation and web search sometimes interpreted as `external_agent` work.

`recall` achieved 100% domain accuracy, but its semantic-request quality and exact-payload transport remained weak: 24% semantic quality and 28% average verbatim recall. That can still hurt Hammer even though the broad domain is correct.

There was one malformed result, `p2-yuki_list-06`, where mandatory reasoning consumed the completion budget and left a truncated JSON object.

## Provider preflight finding

The first smoke used OpenRouter's cheapest eligible provider, Darkbloom. It produced valid JSON for only 3/18 cases because many responses returned hidden reasoning without a final answer. That run was preserved as a provider-failure artifact and was not used for the model comparison.

The controlled DeepInfra smoke produced valid JSON for 18/18 cases and the full run produced 449/450, confirming that provider choice materially affects GPT-OSS structured-output reliability.

## Conclusion

Under Yuki's current semantic-delegation contract, this run does **not** support replacing Qwen3.8-27B with GPT-OSS-20B as the main brain. GPT-OSS is attractive on speed and cost, but its errors occur at important architecture boundaries that restrict which schemas Hammer sees. A wrong Stage A domain can make the correct final tool unavailable downstream.

This does not answer whether GPT-OSS-20B would be a good *direct tool dispatcher*. That would be a separate experiment with the 18 real tool schemas and native tool-call output rather than the semantic delegation contract used here.

## Artifacts

- Full GPT-OSS report: `stage-a/mainbrain-gpt-oss-20b-openrouter-full450-v1-deepinfra.json`
- Controlled smoke: `stage-a/mainbrain-gpt-oss-20b-openrouter-smoke18-v2-deepinfra.json`
- Preserved provider-failure smoke: `stage-a/mainbrain-gpt-oss-20b-openrouter-smoke18-v1.json`
- Machine-readable comparison: `GPT-OSS-20B-VS-QWEN38-STAGE-A-COMPARISON-v1.json`

Full report SHA-256: `f99588031a30cf8fbdbed966c07366fc7fcdeb9e793acadcbdc4c423e2a42cf7`.
