# Hammer 1.5B Pilot v1.1 — Human Review Report

## Decision

**APPROVE FOR FULL HUMAN REVIEW; DO NOT SCALE OR TRAIN YET.** Pilot v1.1 resolves the mechanical weaknesses identified in v1. Its format, annotations, leakage gate, and structured write/append adapter are internally consistent. The 150 records still require human semantic/style approval before they can seed a larger dataset.

No model was loaded or trained. No Yuki tool was executed. The frozen 450 was not run or modified.

Human semantic review produced one accepted correction to the still-unapproved v1.1 artifact. The pre-review dataset checksum was `ea6447c042747f76f1ccc4c6bb47690bc6b0481a1080c5ecdc0a7a92d02aef59`; the corrected checksum is `9bab33fa2e0c1df0701bd51b4da6d8a08940d05d3f88892501ea3ddafa3001ff`. This change is documented below rather than hidden.

## Summary

| Measure | Pilot v1 | Pilot v1.1 |
|---|---:|---:|
| Records | 150 | 150 |
| Strictly valid | 150 | 150 |
| Literal source-span integrity | 86/86 | 86/86 |
| Sacred leakage pass | 149 | **150** |
| Human-review leakage flags | 1 | **0** |
| Leakage rejects in final artifact | 0 | 0 |
| Structured write/append records | 0 | **23** |
| Distinct two-word openings, all records | 109 | **136** |
| Distinct two-word openings, ontology 60 | 39 | **58** |
| Maximum opening repetition | 6 | **3** |

The frozen checksum remains:

```text
5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466
```

## Composition retained

- 60 ontology/tool-confusion records
- 35 literal-preservation records
- 25 semantic-argument records
- 15 ordinary/easy records
- 15 adversarial/edge/rejection records
- 86 `literal_source`
- 54 `semantic_argument`
- 4 `no_argument`
- 6 explicit rejects
- all 18 live Yuki tools represented

Each major ontology family retains exactly 10 records:

- `read` vs `yuki_read`
- `yuki_write` vs `yuki_append`
- `see` vs `image`
- `search` vs `recall`
- `remember` vs `yuki_write`
- `shell` vs `hardware`

## Leakage resolution

The v1 request `Tell me the current local time.` was replaced by:

```text
local clock, rn??
```

The unchanged checker compared all 150 records against all 450 frozen cases. Final result:

```text
pass:   150
review:   0
reject:   0
```

No thresholds were weakened. The exact-match, character, token, trigram, and same-tool literal-payload rules are unchanged.

## Human semantic review repair: `hardware.process` versus `hardware.top`

Human review correctly found that the previous Hammer-visible schema listed both enum values without defining their difference. Inspection of the unchanged live implementation established:

- `process`: Yuki's own current LLM/runtime process only—PID, RSS, CPU usage, and thread count.
- `top`: a system-wide ranking of the top five processes by CPU and the top five by RAM.

The previous `h15-pilot-v1.1-044` wording asked generally which processes were consuming resources but targeted `process`; that was semantically wrong. It is now:

```text
how much CPU and RAM is your own LLM process using?
→ hardware {"metric":"process"}
```

The system-wide control remains:

```text
Give me the busiest processes right now.
→ hardware {"metric":"top"}
```

Both meanings are now stated explicitly in every Hammer-visible hardware schema. No live hardware implementation or production metadata was changed.

## Ontology wording diversity

Pilot v1 used repeated controlled scaffolds. Pilot v1.1 keeps the same labels, arguments, tool subsets, and family balance while rewriting all 60 ontology requests.

Examples:

```text
Need ~/Desktop/Quartz Notes.md; regular filesystem, not your notes.
→ read

not the disk copy—the saved Raven_Index.txt
→ yuki_read

No appending this time; replace colors.cfg with accent=#EF3340
→ yuki_write

don't wipe field-journal.md; tack this on: Add: heard an owl at 02:10.
→ yuki_append

not a new picture; inspect the current view for warning triangle
→ see

need artwork of isometric bakery on a floating pebble
→ image
```

The ontology group now has 58 distinct normalized two-word openings across 60 records. One opening repeats, with a maximum frequency of three. Controlled semantic pairing remains, but opening-token shortcuts are substantially reduced.

## Typo, fragment, and conversational coverage

Human-review tags are audit metadata only:

| Feature | Count |
|---|---:|
| `terse_fragment` | 28 |
| `punctuation_variation` | 24 |
| `negative_contrast` | 16 |
| `shorthand` | 15 |
| `typo` | 14 |
| `casual` | 12 |
| `multi_clause` | 12 |
| `pronoun_context` | 13 |
| `clause_reordering` | 11 |
| `implicit_intent` | 11 |
| `contraction` | 10 |
| `distractor_noun` | 5 |
| `correction` | 2 |

Representative samples:

```text
/opt/local/share/aurora.ini ... read frm the Mac pls
cmd pls -> uptime -p
did i ever tell u about ceramic keyboard caps?
ur file shelf—list it
delete the note—name? forgot it
```

Noise remains a minority and is concentrated in annotated cases rather than applied indiscriminately.

## Other human-review notes

The preservation wording in these two `ask_claude` records is deliberate downstream payload, not a hidden Hammer instruction:

```text
Explain why her value stayed 'null'. Do not change the pronoun.
Compare parse_v2() with parseV2(); preserve both names.
```

In both cases the entire text is intended to reach Claude character-for-character. The surrounding request scaffolding (`External-agent payload:` or `Ask the external agent:`) is excluded from the argument span.

Negative-contrast wording appears in 16/150 records (10.7%). It is useful in this diagnostic pilot, but that proportion should not be copied into a scaled dataset; otherwise Hammer may overfit to words such as “not” instead of learning the positive capability distinction.

Schema exposure is deliberate and matches the intended Qwen → subset → Hammer architecture:

| Offered schemas | Records |
|---:|---:|
| 1 | 9 |
| 2 | 86 |
| 3 | 11 |
| 5 | 27 |
| All 18 | 17 |

Pilot v1.1 is not designed as a predominantly all-18 direct-dispatcher dataset. Any later scaling plan should preserve an explicit exposure policy rather than changing this distribution accidentally.

## Composite write/append repair

Inspection found:

- Live metadata exposes one required `filename_and_content` string.
- Live `split_filename_content()` splits only on the first pipe.
- Pipes inside content therefore survive, but a pipe inside a filename is ambiguous.
- Asking Hammer to build the legacy transport string needlessly mixes two independent literal spans.

Pilot v1.1 exposes `{filename, content}` to Hammer and stores the adapted live shape separately in `runtime_arguments`. All 23 write/append records use the structured form. The edge case with content `left|right` round-trips exactly. A filename containing `|` is rejected before live normalization.

## Validation

All 150 records pass:

- v1.1 JSON Schema validation;
- exact prompt and native-target reproduction;
- structured training-schema validation;
- deterministic structured-to-live adaptation;
- unchanged live-registry validation;
- literal source-span reconstruction;
- semantic contract scoring;
- stored leakage-result reproduction;
- stored validation-result reproduction;
- category, family, feature, and all-tool coverage checks;
- frozen dataset checksum verification.

## Remaining review concerns

1. Linguistic feature tags are hand-authored audit labels, not automatically proven linguistic classifications.
2. Some fragments are intentionally unnatural. Human review should remove examples that feel like generator theater rather than plausible user language.
3. The adapter forbids `|` in Yuki filenames even though production currently sanitizes neither pipes nor this restriction. That is a prototype boundary and must be explicitly adopted if later integrated.
4. `runtime_arguments` is audit-only. Training exporters must never flatten metadata into supervised text.
5. Six rejection examples remain. Review them carefully to avoid teaching the old over-rejection pattern.
6. This pilot still cannot establish model improvement; no inference or training has occurred.

## Recommendation

Approve Pilot v1.1 for line-by-line human review. If reviewers accept the semantic targets and naturalness, freeze its checksums and use its format—not necessarily every sentence—as the basis for a separately planned larger dataset. Do not begin the 6,400-example generation or any fine-tuning merely because mechanical validation passes.
