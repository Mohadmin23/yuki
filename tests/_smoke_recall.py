"""Live end-to-end test of the new /recall flow against deepseek-v4-flash:free.

Two sessions, 4 messages total. Session 1 plants a memory; session 2 asks the
bot to recall it. Logs every OpenRouter HTTP call so we can see which calls
fire (embeddings, chat completions, summarizer).

Run: uv run python tests/_smoke_recall.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import ms_llama
import episodic

MODEL_ID = "openrouter/deepseek/deepseek-v4-flash:free"
SYS_PROMPT = "You are Yuki, a warm helpful AI companion. Keep replies short (1-3 sentences)."


def banner(label):
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)


def run_turn(bot, msg):
    print(f"\nUSER: {msg}", flush=True)
    reply = bot.react_chat(msg)
    print(f"BOT:  {reply}", flush=True)
    ms_llama._cli_record_episode(bot.unlocked_user_id, bot.session_uuid, msg, reply)


banner("SESSION 1 — plant the memory")
bot1 = ms_llama.VoiceChatBot(model_id=MODEL_ID, system_prompt=SYS_PROMPT, tts=None)
print(f"slot={bot1.unlocked_user_id}  session={bot1.session_uuid[:8]}", flush=True)

run_turn(bot1, "Hey Yuki — my granian server keeps crashing on port 8000 whenever I send messages larger than 64KB.")
run_turn(bot1, "I think it's a max body size config issue in granian.toml.")

banner("Manually summarize session 1 (simulating session end)")
s1 = episodic.summarize_session(bot1.unlocked_user_id, bot1.session_uuid)
print(f"SUMMARY 1: {s1}", flush=True)
print(f"episodes stored: {episodic.count_for(bot1.unlocked_user_id)}", flush=True)
slot = bot1.unlocked_user_id
session1_uuid = bot1.session_uuid
del bot1


banner("SESSION 2 — test /recall")
bot2 = ms_llama.VoiceChatBot(model_id=MODEL_ID, system_prompt=SYS_PROMPT, tts=None)
print(f"slot={bot2.unlocked_user_id}  session={bot2.session_uuid[:8]}", flush=True)
assert bot2.session_uuid != session1_uuid, "session_uuid did not rotate!"

run_turn(bot2, "Hey Yuki, do you remember that granian server crash bug I told you about?")
run_turn(bot2, "yeah — what was my theory about the fix?")

banner("Final summary of session 2")
s2 = episodic.summarize_session(bot2.unlocked_user_id, bot2.session_uuid)
print(f"SUMMARY 2: {s2}", flush=True)
print(f"\nDONE. Total episodes for slot: {episodic.count_for(slot)}", flush=True)

# Inspect the DB to confirm both sessions got summaries
db = episodic._get_db()
rows = db.execute(
    "SELECT session_uuid, summary FROM session_summaries WHERE slot_id = ?", (slot,)
).fetchall()
print(f"\nsession_summaries rows for this slot: {len(rows)}")
for uid, summ in rows:
    print(f"  - {uid[:8]}: {summ[:120]}...")
