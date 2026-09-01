# Yuki Tool Dispatcher Lab

An isolated prototype for testing whether a small local model can do exactly one job:

```text
delegated natural-language request -> one structured Yuki tool call
```

It is not a chat interface. It has no conversation history, never receives tool results, and
never generates an assistant answer. Each request causes exactly one dispatcher generation with
a 256-token ceiling, stopped early as soon as one complete JSON object is produced.

The primary benchmark mode uses each checkpoint's trained native call format and normalizes it
internally before validation. A canonical-output mode is included as an explicit A/B experiment.

| Model family | Native call shape |
| --- | --- |
| `Salesforce/xLAM-1b-fc-r` | `{"tool_calls":[{"name":"...","arguments":{...}}]}` |
| `Salesforce/xLAM-2-*` | `[{"name":"...","arguments":{...}}]` |
| `MadeAgents/Hammer2.1-*` | Hammer's fenced tool-call array |

Hammer's exact leading native fence is removed only from parser input so generation can stop at
the completed array. Raw output remains unchanged in reports and the UI. Arbitrary prose,
unknown fences, trailing text, wrong root shapes, and malformed JSON are still rejected.

## Isolation and source of truth

Everything in the experiment lives in `prototypes/tool_dispatcher/`. It imports the current
registry, metadata, schemas, and implementations from `tools/` at runtime. There is no second
tool registry. The category lists only control which live schemas are offered during an
experiment, and startup fails if a category references a tool that no longer exists.

The prototype adapter fixes the current experimental schema gaps without changing Yuki:

- required inputs are derived from the real tool function signatures;
- required strings get `minLength: 1`;
- every argument object gets `additionalProperties: false`;
- `hardware.metric` and `see.target` remain optional;
- `time` and `yuki_list` remain argument-free.

## Prepare the FP16 xLAM model

The official checkpoint is BF16. To run the requested FP16 baseline with MLX, convert it once to
a cache directory outside the repository:

```bash
mkdir -p ~/Library/Caches/yuki-tool-dispatcher
uv run mlx_lm.convert \
  --hf-path Salesforce/xLAM-1b-fc-r \
  --mlx-path ~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16 \
  --dtype float16
```

Model download and conversion require internet access and approximately 3 GB of disk space. The
official xLAM 1B research checkpoint is licensed CC-BY-NC-4.0.

The MLX backend can also load `Salesforce/xLAM-1b-fc-r` directly, but the interface will report
the checkpoint's configured BF16 dtype. Use the converted path when measuring the FP16 question.

The xLAM tokenizer may expose its visible whitespace markers (`Ġ` and `Ċ`) in raw MLX decoder
text. The lab preserves that exact raw generation for debugging, applies only this
xLAM-specific whitespace normalization to the parser input, and shows when normalization was
used. It never strips conversational prose or repairs otherwise malformed output.

The additional local FP16 checkpoints used by the comparison are:

```text
~/Library/Caches/yuki-tool-dispatcher/xLAM-2-1b-fc-r-fp16
~/Library/Caches/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16
~/Library/Caches/yuki-tool-dispatcher/Hammer2.1-3b-fp16
```

Hammer prompts follow the model's reference client convention: the adapted live Yuki schemas are
serialized as JSON inside Hammer's ChatML prompt. They are not passed through the tokenizer's
generic `tools` template, which can render Python representations that are invalid JSON.

## Run the local lab

From the repository root:

```bash
uv run granian \
  --interface asgi \
  --host 127.0.0.1 \
  --port 8787 \
  prototypes.tool_dispatcher.server:app
```

Open `http://127.0.0.1:8787`. Load the MLX model using:

```text
~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16
```

The interface deliberately separates **Route + validate** from **Execute validated call** so a
wrong model decision cannot immediately mutate files or call an external service.

## Run benchmarks

Primary Phase 1 smoke benchmark, all 72 cases and all 18 tools:

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16 \
  --router-mode model_only \
  --output-mode native \
  --output /tmp/yuki-xlam-phase1.json \
  --failures-output /tmp/yuki-xlam-phase1-failures.jsonl
```

Canonical A/B experiment:

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16 \
  --output-mode canonical
```

Subset experiment:

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16 \
  --group information
```

Comparison router modes are `model_only`, `regex_only`, and `regex_then_model`. Tool execution
is off by default and is never part of routing accuracy.

Targeted Hammer failure suite, 96 cases (16 per failure family):

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --cases prototypes/tool_dispatcher/benchmark_hammer_targeted_cases.json \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16 \
  --router-mode model_only \
  --output-mode native \
  --output /tmp/hammer-targeted.json \
  --failures-output /tmp/hammer-targeted-failures.jsonl
```

The six families are weather routing, camera/`see` routing, `read` versus `yuki_read`, search
argument preservation, file-path preservation, and `ask_claude` text preservation.

Oracle schema-subset diagnosis keeps those same cases and uses expected benchmark labels only to
choose which schemas are exposed:

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --cases prototypes/tool_dispatcher/benchmark_hammer_targeted_cases.json \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16 \
  --oracle-profile logical-domains \
  --output /tmp/hammer-logical-oracle.json

uv run python -m prototypes.tool_dispatcher benchmark \
  --cases prototypes/tool_dispatcher/benchmark_hammer_targeted_cases.json \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16 \
  --oracle-profile targeted-confusions \
  --output /tmp/hammer-confusion-oracle.json
```

`remember-vs-recall` is a third oracle profile for the existing eight matching broad cases. These
profiles are intentionally unrealistic benchmark controls and are not automatic domain routers.
The completed analysis is in
`reports/oracle/ORACLE-SUBSET-REPORT.md`.

Compare completed reports without rerunning inference:

```bash
uv run python -m prototypes.tool_dispatcher compare \
  /path/to/xlam-v1-report.json \
  /path/to/xlam-v2-report.json \
  /path/to/hammer-report.json \
  --output /tmp/three-model-comparison.json
```

The comparison command first verifies that every report contains the same requests and expected
calls. It also calculates aggregate and median backend token rates so a one-token reject cannot
distort throughput reporting.

## Argument scoring

Strict argument accuracy remains the primary audit of whether a model rewrote any argument. A
second, conservative execution-equivalent score prevents harmless Unicode, casing, and whitespace
changes from being counted like broken paths or commands.

| Rule | Tools | Equivalent behavior |
| --- | --- | --- |
| No arguments | `time`, `yuki_list` | Exact empty object |
| Normalized text | `weather`, `search`, `hardware`, `see`, `image`, `remember`, `recall`, `ask_claude` | NFKC Unicode normalization, case-folding, and collapsed whitespace |
| Calculation structure | `calc` | Same safely parsed arithmetic AST; no expression evaluation |
| Exact | `fetch`, `read`, `shell`, all Yuki-file tools | Exact keys and values |

This is deliberately deterministic and conservative, not an embedding or LLM semantic judge.
Meaning-preserving paraphrases beyond normalization remain strict failures and can be reviewed
from the saved per-case details.

### Typed transport contract

Later binder experiments proved that execution equivalence and payload transport are separate
questions. The prototype therefore also has a versioned field contract in
`argument-contract-v1.json`:

- `literal_source` fields require an independently annotated, character-exact value copied from
  the immutable raw request;
- `semantic_argument` fields use a deterministic per-field canonicalization rule;
- `no_argument` tools require an empty argument object.

Official strict and execution-equivalent metrics remain unchanged. Contract-aware reports add
coverage and contract accuracy as separate metrics. A missing literal annotation is reported as
unscoreable; the scorer never substitutes old semantic gold and calls it a source span.

An optional annotation sidecar can be passed to the generic benchmark with
`--argument-contract-annotations`. The sidecar must declare
`yuki-argument-contract-v1` and the exact SHA-256 of the case file, so typed annotations can be
added without mutating a frozen dataset.

## Completed smoke results

The corrected native-Hammer runs on the M1 Air (16 GB) produced:

| Dataset | Model | Tool | Exact call | Equivalent call | Schema valid | Malformed | Avg latency | Generated tok/s | Peak MLX |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Broad 72 | Hammer 1.5B FP16 | 91.7% | 87.5% | 88.9% | 94.4% | 0% | 4.69 s | 18.47 | 3.62 GB |
| Broad 72 | Hammer 3B FP16 | 97.2% | 91.7% | 93.1% | 98.6% | 0% | 11.92 s | 9.64 | 6.65 GB |
| Targeted 96 | Hammer 1.5B FP16 | 75.0% | 62.5% | 66.7% | 80.2% | 0% | 4.64 s | 18.44 | 3.62 GB |
| Targeted 96 | Hammer 3B FP16 | 87.5% | 80.2% | 81.3% | 87.5% | 0% | 10.95 s | 9.41 | 6.65 GB |

These are smoke and targeted-diagnostic results, not a reliability conclusion. The runner is
ready for the planned 300-500+ Phase 2 dataset.

## Phase 2 datasets

The included `benchmark_cases.json` is only Phase 1 smoke coverage: four cases per tool. The same
runner accepts a 300-500+ case JSON array through `--cases` without a built-in case ceiling:

```bash
uv run python -m prototypes.tool_dispatcher benchmark \
  --cases /absolute/path/to/phase2-cases.json \
  --backend mlx \
  --model ~/Library/Caches/yuki-tool-dispatcher/xLAM-1b-fc-r-fp16 \
  --output-mode native
```

Each case must include at least:

```json
{
  "request": "What's the weather in Tokyo?",
  "expected_tool": "weather",
  "expected_arguments": {"city": "Tokyo"}
}
```

Reports include tool accuracy, strict argument accuracy, exact-call accuracy, schema validity,
malformed output, latency, token throughput, confusion counts, per-tool failure groups, and JSONL
fine-tuning candidates. External execution failures are reported separately.

## Production-interface delegation experiment

The isolated production-interface path is separate from the older executable prototype:

```text
raw request
  -> main-brain delegation JSON
  -> deterministic domain-to-live-schema mapping
  -> native Hammer generation
  -> parse + validate + score
  -> stop
```

Its `delegation-mainbrain` and `delegation-dispatch` commands expose no execution flag or tool
implementation path. The Stage A report is checksum-frozen before any dispatcher comparison, so
different Hammer models receive the exact same delegation objects. The canonical fields are
`request`, `domain_hint`, and `verbatim`; the main brain never supplies the final Yuki tool name.

The first completed run used **Qwen3-14B-MLX-4bit only as a temporary main-brain surrogate**
because the intended Qwen3.8-27B checkpoint was not local. Its derived source is
`production-delegation-source-v1.json`, built without changing the frozen 450-case Phase 2 file.
The smoke gate selected the predeclared deterministic 180-case fallback. Hammer 1.5B and 3B were
then run separately against the same Stage A SHA-256.

See `reports/production-delegation/PRODUCTION-DELEGATION-REPORT.md` for the result and
`reports/production-delegation/production-delegation-comparison-v1.json` for machine-readable
metrics. Rebuild the comparison and checksum manifest without inference using:

```bash
uv run python -m prototypes.tool_dispatcher.build_production_delegation_report
```

When the intended 27B checkpoint becomes available, rerun Stage A into a new versioned report and
reuse the same Stage B harness. Do not overwrite the surrogate delegation report.

That intended-model rerun is now complete using `qwen/qwen3.8-27b` through OpenRouter and the
full frozen 450-case set. Per the user's updated scope, only Hammer 1.5B was evaluated downstream;
Hammer 3B was not loaded. The run remained stateless and execution-locked. See
`reports/production-delegation/QWEN3.8-27B-HAMMER1.5B-FULL450-REPORT.md` and the adjacent
`qwen3.8-hammer1.5-full450-comparison-v1.json` for the results.

## Safety behavior

- Shell is dry-run unless strict mode is selected and side effects are explicitly allowed.
- Strict shell mode rejects operators, pipes, redirection, substitution, quoting, escapes, globs,
  and commands outside Yuki's existing allowlist before calling the existing shell tool.
- File mutations, memory writes/recall, image generation, camera use, and Claude delegation require
  explicit execution permission.
- Automated benchmarks do not execute tools unless `--execute` is supplied.

## Tests

The prototype tests are mock-based and make no network or external tool calls:

```bash
uv run pytest prototypes/tool_dispatcher/tests -q
```
