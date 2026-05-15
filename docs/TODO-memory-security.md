# TODO: Multi-User Memory Security Rework

**Status:** On hold. Reverted to simple single-user mode via `MULTI_USER_MODE = False` in `ms_llama.py`. All identity/unlock/ownership code is still in the repo — flipping the flag back to `True` restores the multi-user behavior.

**Checkpoint commit:** `cd10122` — "checkpoint: before reverting to simple single-user mode". `git checkout cd10122` to see the last full multi-user state.

---

## 4 design questions to answer before resuming

These are the open questions that blocked the slot-logic rework. No patching until these are resolved — patching symptoms is how we ended up here.

### 1. When do slots get minted?
Current behavior (buggy): the moment `verify_and_advance` sees a non-name fact with zero candidate matches, it mints a fresh slot immediately. That means a returning user saying "I like superman" (not previously recorded) triggers a phantom slot.

Options to consider:
- Require both a name AND a non-name fact before minting
- Require an explicit signal ("I'm new here", "first time")
- Wait N turns of evidence before committing
- Never auto-mint — only mint on an explicit `/remember-me-as` kind of action

### 2. How are name collisions reconciled?
When the user is bound to slot A and says "my name is X", and slot B already has name X stored, what should happen?

Options:
- Refuse the write, prompt Yuki to ask "wait, aren't you the X I know?"
- Auto-merge A into B (or vice versa)
- Keep both, add a "possibly-same-as" pointer
- Log and flag for manual resolution

### 3. How can name-only slots ever re-unlock?
The 2FA floor requires at least one non-name fact match to unlock. Slots with only names stored (jack, alen, ethen, ethan, etc.) are permanently unreachable via the verify hook. This is working-as-designed for security but broken as UX — how does a returning user whose slot only has a name get re-recognized?

Options:
- Seed every new slot with something beyond a name at creation time
- Allow name-only unlock with a cooldown or explicit challenge
- Offer a manual "that's me" button in the UI
- Accept it as a design limitation and bias new-slot creation toward including at least one non-name fact

### 4. Is Yuki ever allowed to hint that other slots exist?
Current `_locked_stub` tells Yuki "you have N known users on file, but you CANNOT see who." This leaks the *existence* of multiple users. And `_narrowing_hint` tells her "N possible people match what's been said."

Options:
- Full isolation: Yuki never knows other slots exist, always treats the current conversation as single-user
- Narrow hints only: Yuki can ask for clarifying details but never implies multiplicity
- Current behavior (leaky but informative)

---

## Bugs still unfixed

### Bug #2: Phantom slot minted too eagerly
**Location:** `ms_llama.py` `verify_and_advance`, the NEW-USER PATH at the bottom of the function.
**Trigger:** Any existing user volunteers a non-name fact that wasn't previously stored in their slot → system assumes they're new and creates a fresh slot.
**Live confirmation:** Trinity Large Preview test session on 2026-04-23 minted `f054ac0d` when "ethan" + "pizza" didn't match existing slots.
**Fix locus:** Rewrite the claim logic per design question #1.

### Bug #3: No name reconciliation after unlock
**Location:** `ms_llama.py` `verify_and_advance`, post-unlock branch (around line 990 in the current state).
**Trigger:** Once a session is bound to slot A, any `('names', X)` fact just appends X to A's names with zero check for whether X already belongs to slot B.
**Observed:** Slot `4682244e` ended up with `names=['jack', 'alen', 'bred']` after a single session where stardust pretended to be different people.
**Fix locus:** Design question #2.

### Bug #4: Name-only slots can never re-unlock
**Location:** `ms_llama.py` `verify_and_advance`, UNLOCK CONDITIONS block (`if len(candidates) == 1 and bot.session_non_name_match`).
**Trigger:** A user whose slot has only names stored (jack, alen, ethen, 'again' before purge) can offer their name perfectly but `session_non_name_match` stays False, so 2FA never fires.
**Observed:** In the Session 2 test, saying "im alen" correctly identified alen's slot as a candidate but never promoted to unlock.
**Fix locus:** Design question #3.

### Bug #6: Confabulation — Yuki invents shared history
**Location:** Memory-injection prompt (`_memory_as_prompt` in `interface/server.py`).
**Trigger:** The unlocked-memory prompt tells Yuki to "reference and build on these facts naturally" but doesn't forbid her from inventing events not in the list.
**Observed:** Trinity Large Preview invented a "pineapple pizza haiku" the user never wrote with her.
**Mitigation shipped:** Added anti-confabulation clause to the `_memory_as_prompt` header in both multi-user and simple-mode variants (2026-04-23). Not a rewrite, but may be enough for simple mode; validate during multi-user rework.

### Bug #7: Chat re-entry (possibly) not hydrating bot.history
**Location:** `interface/server.py` `/api/chats/{id}/load` handler.
**Trigger:** Opening a past chat of the currently unlocked user.
**Observed:** User reported "conversation history doesn't restore" during Trinity test. Not fully traced before revert — code path reads `bot.history = data.get("messages", [])` which LOOKS correct, but something downstream may drop it. Deferred to avoid patching without a clean repro.
**Fix locus:** Trace `bot.history` lifecycle end-to-end; check whether `/chat` handler or the model-call path resets it.

---

## What DID work and should be preserved

These decisions were load-bearing and proved out in testing. The rework should build on them, not redo them.

- **Sticky binding:** once `verify_and_advance` unlocks a slot, candidate narrowing is OFF for the rest of the session. No chance-collision can re-bind. (`verify_and_advance` post-unlock branch.)
- **Chat ownership gate at the API level:** chats stamped with `user_id`, all access endpoints check `_chat_accessible`. Correctly hid other users' chats from the sidebar once memory was locked. Unit tests cover it.
- **UUID-based slots, name-independent identity:** slots are keyed by UUID so renaming / duplicate names don't corrupt identity. The right foundation.
- **Hash-based fact verification:** `_hash_fact(category, value, salt)` with category-scoped SHA-256 means candidate narrowing can happen without exposing plaintext facts to the model pre-unlock. Keep this.
- **2FA floor:** non-name fact required for unlock is the right security idea even though it creates Bug #4. Keep the principle, solve #4 separately.
- **Regression tests for Bug #1 and Bug #5:** `test_ms_llama.py` has locked-in tests for the "my cat name is bred" prompt bug and the "its me again → 'again'" regex bug. Those fixes (prompt patch + regex tightening) apply in both modes and should be kept.
- **Prompt improvements kept in simple mode:** third-person possessive rules in `_EXTRACTION_SYSTEM_PROMPT`, banned-filler expansion in `_rule_based_extract`, anti-confabulation clause in `_memory_as_prompt`.

---

## Files and modules involved

Everything here stays in the repo; simple mode just gates behavior at runtime via `MULTI_USER_MODE`.

### `ms_llama.py`
- `MULTI_USER_MODE` flag (around line 588) — flip to `True` to restore
- `SIMPLE_MODE_PREFERRED_NAME = "stardustv2.0"` — which slot becomes the default in simple mode
- `_default_user_id(store)` — resolves the single-user slot (preferred name → first slot → creates one)
- Memory store helpers: `_normalize_value`, `_hash_fact`, `_empty_store`, `_create_slot`, `_add_fact_to_slot`, `_find_candidates`, `_get_view`, `_load_users`, `_save_users`
- Fact extraction: `_EXTRACTION_SYSTEM_PROMPT`, `_parse_extracted_json`, `_rule_based_extract`, `_extract_from_answer`, `extract_facts`
- `verify_and_advance` — the full claim/unlock/narrow state machine (multi-user only, but still runs; simple-mode auto-binding in `VoiceChatBot.__init__` makes it take the post-unlock path)
- `VoiceChatBot.__init__` — auto-binds to default user when `MULTI_USER_MODE=False`

### `interface/server.py`
- Imports `MULTI_USER_MODE`, `_default_user_id` from ms_llama
- `_memory_as_prompt` — two header variants (simple mode vs multi-user), both include anti-confabulation clause
- `_locked_stub`, `_narrowing_hint` — dead code in simple mode (never reached because the bot is always bound), kept intact for multi-user restore
- `_inject_memory` — unchanged; simple mode auto-binds so the "unlocked" branch always fires
- `_reset_session_lock` — gated: clears state in multi-user mode, re-binds to default user in simple mode
- `_chat_accessible`, `_list_chats` — gated: simple mode returns True / returns all
- `_load_chat` — inherits the gate via `_chat_accessible`
- `/api/memory` GET — always reports `locked: False` in simple mode (because `bot.unlocked_user_id` is always set)
- `/api/memory` POST / DELETE — always succeeds in simple mode (same reason)
- `/api/debug/unlock/*`, `/api/debug/relock` — still wired up, safe to use in either mode for testing

### `test_ms_llama.py`
- 127 tests total (122 pass + 5 pre-existing failures)
- `isolated_memory` fixture — monkeypatches `MEMORY_FILE`/`MEMORY_DIR`/`MEMORY_BACKUP_FILE` to `tmp_path`; reuse for slot-logic rework tests
- `_FakeBot` class — minimal stand-in for extraction/verify tests; extend rather than mock `VoiceChatBot`
- Regression tests tagged with `_bug1_` / `_bug5_` — keep green

---

## Data state at time of revert

**Backups in `data/`:**
- `memory.json.bak-pre-simple-mode` — snapshot right before this revert
- `memory.json.bak-pre-again-purge` — earlier snapshot
- `chats.bak-pre-simple-mode` — chat folder snapshot right before this revert
- `chats.bak-pre-ownership` — pre-ownership-migration snapshot

**Live `data/memory.json` has 8 slots, 1 real + 7 pollution:**
- `330d78c3` → stardustv2.0 (the real you)
- `874dbbad` → jack (test pollution)
- `6bd1c090` → ethan + dragon ball z (test pollution)
- `7d6eac85` → ethen (test pollution)
- `26359f61` → alen (test pollution)
- `4682244e` → phantom with names=[jack, alen, bred] (Bug #2 artifact)
- `0b562459` → ethen + pizza (test pollution)
- `f054ac0d` → ethan + pizza (Bug #2 artifact, Trinity test 2026-04-23)

Simple mode always binds to `330d78c3` (stardustv2.0), so the pollution is invisible to Yuki at runtime. Cleanup deferred to the slot-logic rework per your direction.

---

## How to resume

1. Re-read this file.
2. Answer the 4 design questions.
3. Flip `MULTI_USER_MODE = True` in `ms_llama.py`.
4. Run `./.venv/bin/python -m pytest test_ms_llama.py` — should still be 122/127 passing.
5. Design the new `verify_and_advance` claim/merge/unlock path per answered questions.
6. Clean up the pollution slots (or write a migration that merges ethen/ethan variants).
7. Ship with new regression tests for whatever Bug #2/#3/#4/#7 resolutions you choose.
