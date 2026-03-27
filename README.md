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
# Create environment (Python 3.12 required)
python3.12 -m venv ~/hf-env
c

# Install dependencies
pip install torch transformers bitsandbytes kokoro soundfile mlx-lm fastapi uvicorn
```

## Usage

### CLI Chatbot
Type messages, AI responds with voice. Type `quit` to exit.
```bash
source ~/hf-env/bin/activate
python ms_llama.py
```

### Web Interface
Full UI with animated bubble, word-by-word text, and voice input.
```bash
source ~/hf-env/bin/activate
python interface/server.py
```
Then open **http://localhost:7860**

## Web Interface Features

- **Animated bubble** — audio-reactive bars that pulse with the voice
- **Word animation** — each word blurs in as it's spoken, blurs out after
- **Voice input** — click 🎙️ to enter continuous voice loop (auto-listens after each response)
- **Text input** — type manually and press Enter

## Project Structure

```
llama-voice-assist/
├── ms_llama.py          # CLI voice chatbot
├── interface/
│   ├── server.py        # FastAPI backend (port 7860)
│   └── index.html       # Web UI
└── README.md
```
