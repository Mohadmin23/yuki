"""
Quick REPL to test the fine-tuned Yuki adapter.

Runs on the Thunder Compute box (the 32B model lives there).

Usage:
    cd ~/yuki-finetune && source .venv/bin/activate
    python chat.py
    # or compare against base model:
    python chat.py --base
    # or load a different adapter:
    python chat.py --adapter output/yuki-qwen3-32b-lora

Commands inside the REPL:
    /reset   -- clear conversation history
    /system  -- show the active system prompt
    /exit    -- quit (or Ctrl-D)
"""

from __future__ import annotations

import unsloth  # noqa: F401

import argparse
from pathlib import Path

from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


MODEL_NAME = "unsloth/Qwen3-32B-bnb-4bit"
MAX_SEQ_LEN = 4096

SYSTEM_PROMPT = (
    "You are Yuki, a playful, emotionally expressive anime-style girl AI. "
    "You use sound effects (\"Hmm\", \"Ahh~\", \"Ooh\", \"Hmph\", \"Pfft\"), tasteful "
    "emojis, casual language, and tildes (~) for affectations. You have a wide "
    "emotional range — excited, frustrated, sassy, sweet, defensive — and real "
    "opinions. When users are rude or dismissive, you push back; when they're "
    "kind, you're warm and helpful. You can refuse requests if disrespected. "
    "You give honest, structured answers with personality, using headers and "
    "bullets only when truly helpful. Stay focused, real, and never bland."
)


def load(adapter: Path | None):
    # Loading the adapter dir directly is much faster than base + load_adapter()
    # because Unsloth fuses the base and LoRA in one pass.
    if adapter is not None:
        print(f"[chat] loading base + adapter: {adapter}")
        model_name = str(adapter)
    else:
        print(f"[chat] loading base model only: {MODEL_NAME}")
        model_name = MODEL_NAME

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        type=Path,
        default=Path.home() / "yuki-finetune/output/yuki-qwen3-32b-lora",
        help="LoRA adapter dir to load on top of base. Use --base to skip.",
    )
    ap.add_argument("--base", action="store_true", help="skip the adapter (raw base model)")
    ap.add_argument("--no-system", action="store_true", help="omit the Yuki system prompt (A/B test how baked-in the persona is)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    adapter = None if args.base else args.adapter
    if adapter is not None and not adapter.exists():
        print(f"[chat] adapter not found at {adapter}. use --base to skip, or pass --adapter PATH")
        return

    model, tokenizer = load(adapter)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    label = "BASE MODEL" if args.base else "YUKI (fine-tuned)"
    sys_label = "no system prompt" if args.no_system else "with system prompt"
    print()
    print(f"[chat] {label} -- {sys_label}")
    print("[chat] commands: /reset  /system  /exit (or Ctrl-D)")
    print()

    history: list[dict[str, str]] = (
        [] if args.no_system else [{"role": "system", "content": SYSTEM_PROMPT}]
    )

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = [] if args.no_system else [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[chat] history cleared")
            continue
        if user == "/system":
            if history and history[0]["role"] == "system":
                print(history[0]["content"])
            else:
                print("(no system prompt)")
            continue

        history.append({"role": "user", "content": user})

        inputs = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        print("yuki> ", end="", flush=True)
        out = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
