# Deterministic exact-payload source-span policy v1

Status: frozen before binder implementation.

## Invariant

Every bound string must be copied from one contiguous slice of the immutable raw user request. If a result records offsets `start` and `end`, its value must equal `raw_request[start:end]` exactly.

The binder may use a selected tool, its argument schema, a semantic delegation, model-proposed arguments, and model-provided verbatim hints to identify the intended span. None of those model-produced strings may supply or rewrite the final characters.

## Characters inside a span

Once a span boundary is selected, preserve its characters literally. Do not:

- normalize Unicode;
- change casing or spelling;
- add, remove, or rewrite punctuation;
- add or remove articles, determiners, or pronouns;
- translate or paraphrase;
- alter quotes, backticks, code, paths, symbols, or whitespace inside the span.

Articles and possessives belonging to a content phrase remain part of the payload. For example, the payload in `find my keys` is `my keys`; the payload after `image of a glass city` is `a glass city`. Articles that describe the tool object itself remain surrounding syntax, as in `Create an image of ...`.

## Surrounding syntax

Recognized tool-routing or action syntax is not part of the payload. Examples include `Search for`, `Generate an image of`, `Ask Claude to`, and camera/vision action scaffolding.

A single terminal sentence period or question mark outside an explicit payload boundary is surrounding syntax unless the punctuation changes the transported payload, such as a direct question for an external agent. Explicit `exact`, `intact`, `punctuation`, `prompt:`, `query:`, `question:`, or equivalent payload markers make the marked remainder authoritative.

Outer matching quotes or backticks used only to delimit the entire payload are excluded. Quote or code characters inside a larger payload are preserved. If quoted material is only one component of a longer payload, its delimiters remain part of the payload.

Routing contrast or safety clauses after delimiters such as `; do not`, `; don't`, or `, not` are excluded for `search`, `image`, `see`, and `recall` when they describe which tool must not be used rather than the argument itself. Instructions intended to be delivered to `ask_claude`, including safety constraints on what Claude should not do, remain part of that message.

Generic context nouns such as `discussion`, `conversation`, `topic`, or `memory` may be surrounding retrieval syntax for `recall`. Their content complement is the literal topic span. This policy does not synthesize reordered topic names.

## Ambiguity and abstention

If more than one source boundary remains plausible after applying the general rules and semantic hints, the result is `ambiguous`. Record every candidate span and the reason. Do not choose one silently.

If no defensible source span exists, the result is `unbound`. Do not fall back to generated text.

An ambiguous or unbound required argument yields no bound final call. Optional arguments may remain absent only when the raw request does not identify a target. A source request containing a plausible but ambiguous optional target must still abstain rather than silently emit an empty argument object.

## Tool-specific scope for this experiment

- `search.query`: preserve the complete search expression after routing syntax and before a routing guard.
- `image.prompt`: preserve the complete content description after generation syntax and before a routing guard.
- `ask_claude.question`: preserve the complete message or question intended for Claude, including internal punctuation and intended safety instructions.
- `see.target`: preserve the complete visual target noun phrase, including its content determiner or possessive, but not camera scaffolding or locative/routing instructions.
- `recall.topic`: preserve the literal topic phrase after memory-retrieval syntax; never synthesize a normalized topic that does not occur in the raw request.

## Scoring separation

The original benchmark gold is immutable. Report official exact accuracy separately as baseline versus binder.

Literal-source exact accuracy uses the independently frozen annotations governed by this policy. Ambiguous annotations are reported and excluded from conditional literal-reference accuracy, while conservative end-to-end literal accuracy counts abstention as unsuccessful. Also report the raw-slice invariant rate for every emitted bound string.

The binder implementation must not import, read, receive, or derive from the literal annotation file or official benchmark gold. Annotation data is available only to the evaluation/scoring layer after binder predictions have been frozen.
