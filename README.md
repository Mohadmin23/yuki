# llama-voice-assist

A local voice assistant running entirely on your machine using Qwen2.5-1.5B (MLX) + Kokoro TTS.

## Stack

| Component | Model |
|-----------|-------|
| LLM | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| TTS | `hexgrad/Kokoro-82M` (voice: `af_heart`) |
| Python | 3.12 (required for Kokoro) |

## Setup

```bash
llama setup
```

This creates a Python 3.12 venv and installs all dependencies automatically.

## Usage

The `llama` CLI can be run from any directory.

### CLI Chatbot
Type messages, AI responds with voice. Type `quit` to exit.
```bash
llama cli
```

### Web Interface
Full UI with animated bubble, word-by-word text, and voice input.
```bash
llama web
```
Then open **http://localhost:7860**

### Run Tests
```bash
llama test
```

### All Commands
```
llama setup   — Create Python 3.12 venv and install all dependencies
llama cli     — Launch the terminal chatbot (select model, persona, TTS)
llama web     — Launch the web UI with animated bubble on port 7860
llama test    — Run the test suite with pytest
llama help    — Show help
```

## Features

- **Token stats** — shows tokens/s after every LLM response
- **MLX warmup** — pre-compiles the compute graph so first response is fast
- **Multi-backend** — supports GGUF (llama-cpp), MLX (Apple Silicon), and Transformers
- **Auto model discovery** — searches external drive and `~/.cache/huggingface/hub`
- **Personas** — choose a character (Yuki, Coder, Therapist) or go default
- **ReAct tools** — LLM can search the web, check weather/time, read/write files, run safe shell commands
- **Autonomous mode** — bot speaks unprompted after silence

## Web Interface Features

- **Animated bubble** — audio-reactive bars that pulse with the voice
- **Word animation** — each word blurs in as it's spoken, blurs out after
- **Voice input** — click the mic to enter continuous voice loop (auto-listens after each response)
- **Text input** — type manually and press Enter

## Project Structure

```
llama-voice-assist/
├── llama                # CLI launcher (symlinked to PATH)
├── pyproject.toml       # Python deps (managed with uv)
├── ms_llama.py          # Core voice chatbot engine
├── episodic.py          # Episodic vector memory (sqlite-vec + OpenRouter)
├── interface/
│   ├── server.py        # FastAPI backend (port 7860)
│   └── *.html           # Web UI
├── personas/
│   ├── yuki.txt         # Playful anime character
│   ├── coder.txt        # Senior engineer persona
│   └── therapist.txt    # Empathetic therapist persona
├── data/                # Runtime state (memory.json, episodic_v2.db, chats/)
│   └── archive/         # Old memory backups
├── tests/               # Pytest suite (run with `llama test`)
├── docs/                # Design notes (TODO-memory-security, system-note)
├── subprojects/         # Side workflows kept out of the main tree
│   ├── img2img/         # FLUX Kontext + Counterfeit refine
│   ├── finetune/        # LoRA training pipeline (Thunder Compute)
│   └── Fine-tune-Dataset-Quality-Scorer/
└── yuki/                # Bot's personal file storage (images, scratch)
```
