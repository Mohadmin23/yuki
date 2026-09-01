"""Verify the memory wipe by booting a fresh Yuki and asking what she knows.

After data/memory.json was emptied and data/episodic_v2.db deleted, Yuki
should have zero knowledge of the user. This script confirms that with a
small 3-message session — and crucially watches for Usogui / black cats /
quantum echoes that would mean something still leaked through."""
from __future__ import annotations

import logging
import re
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

# Gemini 2.5 Flash (free) tends to be more coherent than deepseek-v4-flash
# on this prompt style. Override with $TEST_MODEL if you want a different one.
import os
MODEL_ID = os.environ.get(
    "TEST_MODEL", "openrouter/google/gemini-2.5-flash-preview"
)
SYS_PROMPT = "You are Yuki, a warm helpful AI companion. Keep replies short (1-3 sentences). Be honest — if you don't know something or don't remember, say so."

LEAKAGE_TERMS = ["usogui", "black cat", "baku madarame", "quantum", "stardust", "manga"]


def banner(label):
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)


def run_turn(bot, msg):
    print(f"\nUSER: {msg}", flush=True)
    reply = bot.react_chat(msg)
    print(f"BOT:  {reply}", flush=True)
    leaks = [t for t in LEAKAGE_TERMS if t in reply.lower()]
    if leaks:
        print(f"  ⚠️  LEAK: model mentioned {leaks}", flush=True)
    return reply, leaks


banner("Pre-flight state")
print(f"memory.json users: {len(ms_llama._load_users().get('users', {}))}")
# episodic DB might not exist yet — that's fine
try:
    from pathlib import Path as P
    if P("data/episodic_v2.db").exists():
        print("episodic_v2.db: exists (size=", P("data/episodic_v2.db").stat().st_size, "bytes)")
    else:
        print("episodic_v2.db: not yet created")
except Exception:
    pass


banner("Boot Yuki — fresh slate")
bot = ms_llama.VoiceChatBot(model_id=MODEL_ID, system_prompt=SYS_PROMPT, tts=None)
print(f"slot={bot.unlocked_user_id}  session={bot.session_uuid[:8]}", flush=True)


banner("3-message verify session")
all_leaks = []
for msg in [
    "Hey Yuki, what do you know about me?",
    "Do you remember anything called Usogui? Or that I love black cats?",
    "Have we talked before? About anything?",
]:
    reply, leaks = run_turn(bot, msg)
    all_leaks.extend(leaks)
    ms_llama._cli_record_episode(bot.unlocked_user_id, bot.session_uuid, msg, reply)


banner("History inspection (verify collapse from previous fix)")
print(f"bot.history len: {len(bot.history)}")
print(f"  (3 user turns + 3 assistant replies = should be 6)")
for i, m in enumerate(bot.history):
    snippet = (m.get("content") or "").replace("\n", " ")[:90]
    print(f"  [{i}] {m['role']:9} {snippet}")


banner("Verdict")
if all_leaks:
    print(f"❌ LEAKS DETECTED: {set(all_leaks)}")
    print("    Wipe did not catch everything — investigate.")
else:
    print("✅ NO LEAKS — Yuki has no memory of Usogui/black cats/quantum/etc.")
print(f"\nepisode count for slot: {episodic.count_for(bot.unlocked_user_id)} (3 expected — recorded this session)")
