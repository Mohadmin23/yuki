# Hammer 1.5B Pilot v1.2 — Final Human-Approval Report

## Final decision

**APPROVE FOR STAGED SCALING**

This approval means only that the record format and the 150-record pilot are approved as design/reference artifacts. A later task may design and generate the first 500-record staged training batch.

It does **not** authorize model training, model inference, a frozen-450 run, Yuki tool execution, a 2,000-record stage, or automatic 5,000/700/700 generation.

Pilot v1.2 is now immutable. Any later content change must use a new version.

## Human-review history

| Verdict | Count |
|---|---:|
| KEEP | 145 |
| REVISE | 5 |
| REJECT | 0 |
| Final records | 150 |

The source was checksum-locked corrected Pilot v1.1. A deterministic comparison confirmed that semantic/model-visible content changed for exactly records 030, 031, 036, 094, and 117. Version, provenance, validation, and record-ID metadata changed across all rows as required for the new immutable version.

## Five human revisions

### 030 — `ask_claude` literal punctuation

```text
Ask Claude this exact question: Can `map[key]` mutate when key == 'A/B'?
```

Target:

```json
{"question":"Can `map[key]` mutate when key == 'A/B'?"}
```

The literal payload ends with exactly one question mark. The wrapper adds no period. The source span reproduces all 40 payload characters exactly.

### 031 — `image.prompt` punctuation boundary

```text
Create new artwork from this exact prompt: tiny robot holding a sign: DON'T PANIC
```

Target:

```json
{"prompt":"tiny robot holding a sign: DON'T PANIC"}
```

No sentence-ending period is accidentally attached to the artwork/sign text. The prompt remains literal-source.

### 036 — `remember.fact` semantic assertion

Request retained:

```text
dont forget this abt me: no meetings before ten
```

Canonical semantic target:

```json
{"fact":"user does not want meetings before ten"}
```

This preserves the negative preference explicitly. `remember.fact` remains `semantic_argument`; no literal span was introduced.

### 094 — `see.target` referent

```text
check the blue toolbox in the current view
```

Target:

```json
{"target":"blue toolbox"}
```

The request now clearly asks to inspect the toolbox itself, not the area around it. `see.target` remains semantic.

### 117 — `shell.command` naturalness

```text
Need the line count, not system stats—run exactly: wc -l changelog.txt
```

Target:

```json
{"command":"wc -l changelog.txt"}
```

The command is copied character-for-character from its source span. The wording preserves the shell-versus-hardware boundary without generator-like scaffolding.

## Final composition

### Argument modes

| Mode | Count |
|---|---:|
| `literal_source` | 86 |
| `semantic_argument` | 54 |
| `no_argument` | 4 |
| Explicit reject | 6 |

### Categories

| Category | Count |
|---|---:|
| Ontology/confusion | 60 |
| Literal preservation | 35 |
| Semantic argument | 25 |
| Ordinary/easy | 15 |
| Adversarial/edge/rejection | 15 |

### Tools

| Tool | Count | Tool | Count |
|---|---:|---|---:|
| `ask_claude` | 7 | `calc` | 6 |
| `fetch` | 4 | `hardware` | 11 |
| `image` | 12 | `read` | 11 |
| `recall` | 10 | `remember` | 10 |
| `search` | 12 | `see` | 11 |
| `shell` | 9 | `time` | 2 |
| `weather` | 6 | `yuki_append` | 9 |
| `yuki_delete` | 2 | `yuki_list` | 2 |
| `yuki_read` | 6 | `yuki_write` | 14 |
| Reject/no tool | 6 |  |  |

All 18 live Yuki tools remain represented.

### Ontology families

Each family retains exactly 10 records:

- `read` vs `yuki_read`
- `yuki_write` vs `yuki_append`
- `see` vs `image`
- `search` vs `recall`
- `remember` vs `yuki_write`
- `shell` vs `hardware`

## Validation and leakage

| Check | Result |
|---|---:|
| JSON/schema validation | 150/150 |
| Hammer-native target reproduction | 150/150 |
| Live-schema validation | 150/150 |
| Typed argument validation | 150/150 |
| Structured write/append adapter | Passed |
| Literal source-span integrity | 86/86 |
| Semantic target validation | 54/54 |
| Rejections | 6 |
| Leakage pass | 150 |
| Leakage review | 0 |
| Leakage reject | 0 |

Leakage thresholds were not changed. Every record was rechecked deterministically against all 450 frozen requests.

## Architecture notes retained from review

The preservation instructions in the reviewed Claude records are intentionally part of the literal message delivered to Claude:

- `Explain why her value stayed 'null'. Do not change the pronoun.`
- `Compare parse_v2() with parseV2(); preserve both names.`

Negative-contrast language remains useful in this diagnostic pilot, but its current ratio must not be copied automatically into a scaled dataset. A larger batch should prevent Hammer from depending on the token “not” instead of positive capability semantics.

Schema exposure remains deliberate for the intended Qwen → subset → Hammer architecture:

| Offered tools | Records |
|---:|---:|
| 1 | 9 |
| 2 | 86 |
| 3 | 11 |
| 5 | 27 |
| All 18 | 17 |

Pilot v1.2 is not primarily a direct-all-18 dispatcher dataset.

## Tests and safety

- Prototype tests: **253 passed**.
- Dispatcher lint: **passed**.
- Models loaded: **none**.
- Model inference or training: **none**.
- Frozen-450 benchmark runs: **none**.
- Yuki tools executed: **none**.
- Production Yuki behavior changed: **no**.

## Frozen hashes

```text
Original Pilot v1:
563f4cc6f159f497836adaa585f8b3fd437a20126310541a8427f19f878a3c82

Corrected Pilot v1.1 source:
9bab33fa2e0c1df0701bd51b4da6d8a08940d05d3f88892501ea3ddafa3001ff

Approved Pilot v1.2:
6ba64cde883530ffb137cd9747cbe79d1c8a85686822d8c0d65da4758cb8c459

Frozen phase2-cases.json:
5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466
```

The approved v1.2 dataset must not be edited in place.
