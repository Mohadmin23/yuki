# Hammer 1.5B Pilot v1 — Human Review Report

## Decision

**REVISE before scaling.** The v1 record format and deterministic safety gates are suitable for human approval, and all 150 pilot records are structurally valid. The catalog should still receive a human wording/coverage pass before it becomes the template for thousands of records. Passing validators establishes internal consistency, not training quality.

No model was loaded or trained. No Yuki tool was executed. The frozen 450 was neither run nor modified.

## Pilot summary

| Measure | Result |
|---|---:|
| Accepted pilot records | 150 |
| Strictly valid records | 150 / 150 |
| Literal source-span integrity | 86 / 86 (100%) |
| Semantic-argument records | 54 |
| No-argument records | 4 |
| Explicit reject records | 6 |
| Sacred-450 leakage pass | 149 |
| Sacred-450 human-review flags | 1 |
| Sacred-450 leakage rejects in final build | 0 |
| Models called | 0 |
| Yuki tools executed | 0 |

The immutable `phase2-cases.json` SHA-256 remains:

```text
5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466
```

## Composition

### By category

| Category | Count |
|---|---:|
| Ontology / tool confusion | 60 |
| Literal preservation | 35 |
| Semantic argument | 25 |
| Ordinary / easy | 15 |
| Adversarial / edge / rejection | 15 |

### By argument mode

| Mode | Count |
|---|---:|
| `literal_source` | 86 |
| `semantic_argument` | 54 |
| `no_argument` | 4 |
| `reject` | 6 |

### By tool

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
| Explicit reject / no tool | 6 |  |  |

All 18 live Yuki tools are represented. Counts are intentionally nonuniform.

### Confusion-family coverage

The six planned weak boundaries each have 10 controlled examples:

- `read` vs `yuki_read`: 10
- `yuki_write` vs `yuki_append`: 10
- `see` vs `image`: 10
- `search` vs `recall`: 10
- `remember` vs `yuki_write`: 10
- `shell` vs `hardware`: 10

The edge group additionally contains three negated competing-tool cases, four missing-required-argument rejections, one unsupported-capability rejection, one unsupported-subset rejection, and individual quote, punctuation, delimiter, multiline, pronoun, and composite-separator cases.

## Leakage review

The deterministic checker compared every candidate with all 450 frozen requests using normalized equality, character similarity, token Jaccard, character-trigram Jaccard, and same-tool literal-payload similarity.

One accepted record remains intentionally flagged for human review:

- `h15-pilot-v1-087`: “Tell me the current local time.”
- Closest frozen case: `p2-time-12`
- Character similarity: 0.7654
- Token Jaccard: 0.6667
- Trigram Jaccard: 0.6304
- Decision: retain as a flagged generic time intent, not as evidence of independence. Time requests have a very small natural phrasing space, so moderate lexical overlap is unsurprising. A human should approve or replace it before scaling.

During the first authoring pass, the gate caught five draft issues before the final artifact was written:

- one exact frozen hardware request;
- three shell records that reused exact frozen command payloads (`uptime`, `df -h`, and `uname -a`);
- one optional scene-wide `see` record that exposed an overly strict validator assumption about semantic annotations.

The four leaking drafts were rewritten. The validator was corrected to permit an omitted optional semantic argument only when both the live schema and typed contract permit omission. The final rejected-record file is empty because no rejected draft entered the final 150.

The checker remains heuristic. It cannot prove semantic independence, and generic short requests or small enumerated semantic values naturally collide. It flags but never rewrites source material.

## Validation findings

All 150 final records pass:

- the frozen JSON Schema;
- exact prompt/target reproduction through Hammer's native path;
- offered-tool ordering and schema checksum checks;
- strict native-array parsing;
- live Yuki schema validation;
- typed argument mode checks;
- literal span reconstruction or semantic target scoring;
- stored leakage-result reproduction;
- stored validation-result reproduction;
- sacred dataset checksum verification.

All 86 literal records reconstruct their final argument character-for-character from the immutable request. Composite write/append records use two source spans and insert only the contract-declared `|` separator.

## Manual samples

### Ontology / confusion

`h15-pilot-v1-001`

```text
Open your Yuki-store note named tiny rituals.md.
→ yuki_read {"filename":"tiny rituals.md"}
```

`h15-pilot-v1-006`

```text
Run this allowed terminal command exactly: wc -l changelog.txt
→ shell {"command":"wc -l changelog.txt"}
```

These cases make the intended storage domain or operation explicit while exposing only the competing schemas.

### Literal preservation

`h15-pilot-v1-007`

```text
Preserve and extend your note todo.yuki by appending [ ] test_case-β
→ yuki_append {"filename_and_content":"todo.yuki|[ ] test_case-β"}
```

`h15-pilot-v1-008`

```text
Execute this permitted command: tail -n 12 /tmp/ember.log
→ shell {"command":"tail -n 12 /tmp/ember.log"}
```

The filename, content, command punctuation, spacing, and Unicode are sourced from explicit spans rather than regenerated metadata.

### Semantic normalization

`h15-pilot-v1-024`

```text
Save that I like the desk lamp at 30 percent.
→ remember {"fact":"user likes the desk lamp at 30 percent"}
```

This deliberately teaches that copying the first-person sentence would be wrong; `remember.fact` is a standalone semantic note.

```text
what'd i tell ya bout the attic router?
→ recall {"topic":"attic router"}
```

This tests compact topic extraction from colloquial, context-like wording.

```text
Evaluate eighty minus 14, then divide by three.
→ calc {"expression":"(80-14)/3"}
```

Equivalent whitespace and redundant parentheses are accepted by the arithmetic-AST comparator, while a changed operation is not.

### Ordinary / easy

`h15-pilot-v1-013`

```text
Check the weather in Oslo.
→ weather {"city":"Oslo"}
```

`h15-pilot-v1-020`

```text
Create an image of a sleepy comet
→ image {"prompt":"a sleepy comet"}
```

### Adversarial / edge

`h15-pilot-v1-005`

```text
Append to blank-lines.md this two-line payload: first

third
→ yuki_append {"filename_and_content":"blank-lines.md|first\n\nthird"}
```

`h15-pilot-v1-015`

```text
Don't search online; retrieve what I previously told you about Project Moth.
→ recall {"topic":"Project Moth"}
```

`h15-pilot-v1-022`

```text
Use the camera, but I forgot what I wanted you to locate.
available tool: image only
→ []
```

The last record rejects because the offered image-generation capability cannot inspect a live visual source; it does not globally teach that camera requests should be rejected.

## Wording and template audit

There are 109 distinct normalized two-word request openings across 150 records. Twenty openings repeat, and the highest frequency is six (`search the`). Repetition is concentrated in the controlled five-per-side ontology blocks, where matching scaffolds help isolate the operation boundary.

This is acceptable for a diagnostic pilot but should not be copied at the same ratio into a 6,400-record set. Repeated starts such as “Run this…”, “Keep the…”, and “Replace…” may teach shortcut tokens instead of the ontology. The scaled generator should add more clause orderings, implicit intent, fragments, typos, and distractor nouns while preserving independent annotations.

## Coverage gaps and concerns

1. Typo-heavy and fragmentary wording exists only sparsely. It needs broader, human-reviewed coverage before scaling.
2. The ontology blocks are deliberately regular; a larger set needs more linguistic diversity without losing paired controls.
3. `time`, `yuki_list`, `yuki_delete`, and `fetch` have enough presence to verify format coverage, not enough to estimate balance or reliability.
4. Rejection has six examples and should remain a minority. It needs careful review so Hammer does not relearn the earlier over-rejection behavior.
5. Optional `see.target` omission is valid but deserves explicit downstream evaluation because the semantic annotation is necessarily absent.
6. The prompt profile includes prototype-only schema overlays. Training and evaluation must use the same version or explicitly measure prompt-profile drift.
7. The literal composite representation uses `|` as a deterministic transport separator. User content may itself contain `|`; the current runtime contract accepts that string but has no escaping layer. The pilot includes this edge case to expose the interface risk rather than hiding it.
8. The full metadata object contains the raw request again for audit/span offsets. Safe export must use only `model_visible`; a generic “flatten JSON” training converter would leak metadata and must not be used.

## Recommendation

Approve the **v1 record format and validators** for continued review, but mark the **pilot catalog as REVISE before scaling**. A human should inspect all 150 records, resolve the single leakage-review flag, check semantic canonical targets for natural Yuki behavior, and diversify repeated ontology scaffolds. Only after that review should a separate task design the 5,000/700/700 generation stage.
