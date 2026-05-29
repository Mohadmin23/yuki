"""
TUI to test the fine-tuned Yuki adapter.

Full-screen prompt_toolkit interface: a scrolling conversation pane on top, a
divider line, and a fixed input line anchored at the bottom. Replaces the old
input()/TextStreamer REPL, which leaked terminal escape codes and gave no
separation between what you typed and Yuki's replies.

Runs on the Thunder Compute box (the 32B model lives there).

Usage:
    cd ~/yuki-finetune && source .venv/bin/activate
    python chat.py
    # compare against the raw base model:
    python chat.py --base
    # load a different adapter:
    python chat.py --adapter output/yuki-qwen3-32b-lora
    # A/B how baked-in the persona is (no system prompt):
    python chat.py --no-system

Keys / commands:
    Enter             send message
    Ctrl-C / Ctrl-D   quit
    /reset            clear conversation history
    /system           show the active system prompt
    /exit             quit
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before torch)

import argparse
import asyncio
import threading
from pathlib import Path

from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea


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


class _UIStreamer(TextStreamer):
    """TextStreamer that routes decoded chunks to a callback instead of stdout.

    The base class prints to stdout, which would corrupt a full-screen TUI. We
    override on_finalized_text so every chunk goes to the UI marshalling fn.
    """

    def __init__(self, tokenizer, on_text):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._on_text = on_text

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._on_text(text)


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

    def initial_history() -> list[dict[str, str]]:
        return [] if args.no_system else [{"role": "system", "content": SYSTEM_PROMPT}]

    history = initial_history()

    # ---- shared state (mutated only on the UI/event-loop thread) ----
    state = {"transcript": "", "generating": False}
    captured: dict[str, asyncio.AbstractEventLoop] = {}

    # ---- widgets ----
    conversation = TextArea(
        text="",
        read_only=True,
        scrollbar=True,
        wrap_lines=True,
        focusable=False,
    )
    divider = Window(height=1, char="─", style="class:divider")
    input_field = TextArea(
        height=1,
        prompt="you> ",
        multiline=False,
        wrap_lines=False,
    )

    label = "BASE MODEL" if args.base else "YUKI (fine-tuned)"
    sys_label = "no system prompt" if args.no_system else "with system prompt"
    status = Window(
        content=FormattedTextControl(
            f" {label} · {sys_label} · Enter=send  /reset  /system  /exit  (Ctrl-C quits)"
        ),
        height=1,
        style="class:status",
    )

    def render():
        # set_document with the cursor at the end keeps the view auto-scrolled.
        text = state["transcript"]
        conversation.buffer.set_document(Document(text, len(text)), bypass_readonly=True)
        app.invalidate()

    def ui_append(text: str):
        state["transcript"] += text
        render()

    def from_thread(fn, *fn_args):
        # Worker threads must not touch prompt_toolkit state directly; hop back
        # onto the event loop thread.
        captured["loop"].call_soon_threadsafe(fn, *fn_args)

    def generate(user_text: str):
        history.append({"role": "user", "content": user_text})
        inputs = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 is a hybrid reasoning model; Yuki's
            # dataset has no <think> blocks, so disable thinking or the template
            # injects scaffolding that leaks into replies.
            return_tensors="pt",
        ).to(model.device)

        from_thread(ui_append, "\nyuki> ")
        streamer = _UIStreamer(tokenizer, lambda t: from_thread(ui_append, t))
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

        def done():
            state["transcript"] += "\n"
            state["generating"] = False
            render()

        from_thread(done)

    def on_enter(buff) -> bool:
        if state["generating"]:
            return True  # busy; keep the typed text, do nothing

        text = buff.text.strip()
        if not text:
            return False
        if text == "/exit":
            app.exit()
            return False
        if text == "/reset":
            history[:] = initial_history()
            ui_append("\n[history cleared]\n")
            return False
        if text == "/system":
            if history and history[0]["role"] == "system":
                ui_append(f"\n[system] {history[0]['content']}\n")
            else:
                ui_append("\n[system] (no system prompt)\n")
            return False

        ui_append(f"\nyou> {text}")
        state["generating"] = True
        render()
        threading.Thread(target=generate, args=(text,), daemon=True).start()
        return False  # clear the input line

    input_field.accept_handler = on_enter

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("c-d")
    def _(event):
        event.app.exit()

    root = HSplit([status, conversation, divider, input_field])
    style = Style.from_dict({
        "status": "reverse",
        "divider": "#666666",
    })
    app = Application(
        layout=Layout(root, focused_element=input_field),
        key_bindings=kb,
        style=style,
        full_screen=True,
        mouse_support=True,
    )

    def pre_run():
        captured["loop"] = asyncio.get_running_loop()

    app.run(pre_run=pre_run)


if __name__ == "__main__":
    main()
