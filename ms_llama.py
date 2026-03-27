import os
import sys
import subprocess
import readline
import numpy as np
import soundfile as sf
from pathlib import Path

# Disable readline bell (triggers on long paste)
readline.parse_and_bind('set bell-style none')

# ============= MODEL SELECTION =============

def select_model(hub_dir="/Volumes/madisk/huggingface/hub"):
    hub = Path(hub_dir)
    if not hub.exists():
        raise FileNotFoundError(f"Hub directory not found: {hub_dir}")

    model_dirs = sorted([d for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")])

    entries = []
    for d in model_dirs:
        parts = d.name.split("--")
        model_name = "/".join(parts[1:])
        gguf_files = sorted(d.rglob("*.gguf"))
        if gguf_files:
            for f in gguf_files:
                entries.append((str(f), f"{model_name}  [{f.name}]"))
        else:
            entries.append((model_name, model_name))

    if not entries:
        raise RuntimeError(f"No models found in {hub_dir}")

    print("\nAvailable models:")
    for i, (_, label) in enumerate(entries):
        print(f"  [{i + 1}] {label}")

    while True:
        try:
            choice = int(input("\nSelect model number: ")) - 1
            if 0 <= choice < len(entries):
                model_id, label = entries[choice]
                print(f"Selected: {label}\n")
                return model_id
            print("Invalid choice, try again.")
        except ValueError:
            print("Invalid choice, try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

# ============= PERSONALITY SELECTION =============

def select_personality():
    try:
        import termios
        _has_termios = True
    except ImportError:
        _has_termios = False

    answer = input("Do you want to add a personality? (yes/no): ").strip().lower()
    if answer in ("yes", "y"):
        personality = input("Describe the personality: ").strip()
        if personality:
            if _has_termios and os.isatty(0):
                termios.tcflush(0, termios.TCIFLUSH)
            print(f"Personality set: {personality}\n")
            return personality
    if _has_termios and os.isatty(0):
        termios.tcflush(0, termios.TCIFLUSH)
    return None

# ============= TERMINAL PROTECTION =============

class _SuppressIO:
    """Redirects stdin/stdout/stderr to /dev/null and restores terminal on exit."""

    def __enter__(self):
        self._saved_term = None
        self._old = None
        self._devnull_r = None
        self._devnull_w = None
        try:
            import termios
            if os.isatty(0):
                self._saved_term = termios.tcgetattr(0)
        except Exception:
            pass
        try:
            self._devnull_r = open(os.devnull, "r")
            self._devnull_w = open(os.devnull, "w")
            self._old = (os.dup(0), os.dup(1), os.dup(2))
            os.dup2(self._devnull_r.fileno(), 0)
            os.dup2(self._devnull_w.fileno(), 1)
            os.dup2(self._devnull_w.fileno(), 2)
        except Exception:
            self._cleanup_devnull()
        return self

    def __exit__(self, *_):
        if self._old is not None:
            for i, fd in enumerate(self._old):
                try:
                    os.dup2(fd, i)
                    os.close(fd)
                except Exception:
                    pass
        self._cleanup_devnull()
        if self._saved_term is not None:
            try:
                import termios
                termios.tcsetattr(0, termios.TCSANOW, self._saved_term)
            except Exception:
                pass

    def _cleanup_devnull(self):
        for f in (self._devnull_r, self._devnull_w):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

# ============= LLM SETUP =============

def setup_llm(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", cache_dir=None):
    if cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    if str(model_id).endswith(".gguf"):
        model_path = Path(model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {model_id}")
        from llama_cpp import Llama
        with _SuppressIO():
            model = Llama(model_path=str(model_id), n_ctx=4096, n_gpu_layers=-1, verbose=False)
        print("  Backend: llama-cpp (GGUF)")
        return model, None, "gguf"

    try:
        from mlx_lm import load
        model, tokenizer = load(model_id)
        print("  Backend: MLX")
        return model, tokenizer, "mlx"
    except ImportError:
        print("  MLX not installed, falling back to transformers...")
    except Exception as e:
        print(f"  MLX failed ({e}), falling back to transformers...")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device, cache_dir=cache_dir
        )
    except Exception:
        print("  MPS failed, retrying on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cpu", cache_dir=cache_dir
        )
    print(f"  Backend: transformers ({model.device})")
    return model, tokenizer, "transformers"

# ============= KOKORO TTS SETUP =============

def setup_kokoro_tts():
    from kokoro import KPipeline
    return KPipeline(lang_code='a')

# ============= VOICE CHATBOT =============

MAX_HISTORY_TURNS = 10
TTS_MAX_CHARS = 500
MAX_INPUT_CHARS = 2000

class VoiceChatBot:
    def __init__(self, model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", cache_dir=None, system_prompt=None):
        print(f"🚀 Loading {model_id}...")
        self.llm_model, self.llm_tokenizer, self.backend = setup_llm(model_id, cache_dir)

        print("🎵 Setting up Kokoro TTS...")
        self.tts = setup_kokoro_tts()

        self.system_prompt = system_prompt
        self.history = []
        print("✅ Ready! Type your message and press Enter.\n")

    def _trim_history(self):
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def chat(self, user_input: str, max_tokens: int = 1024) -> str:
        user_input = user_input[:MAX_INPUT_CHARS]
        self.history.append({"role": "user", "content": user_input})

        if self.backend == "gguf":
            # Build messages: inject system prompt into first user message
            # since many GGUF models don't support the system role
            messages = []
            first_user = True
            for msg in self.history:
                if msg["role"] == "user" and first_user and self.system_prompt:
                    messages.append({"role": "user", "content": f"{self.system_prompt}\n\n{msg['content']}"})
                    first_user = False
                else:
                    messages.append(msg)
                    if msg["role"] == "user":
                        first_user = False
            with _SuppressIO():
                raw = self.llm_model.create_chat_completion(messages=messages, max_tokens=max_tokens)
            try:
                response = raw["choices"][0]["message"]["content"].replace("\a", "")
            except (KeyError, IndexError):
                response = "Sorry, I could not generate a response."

        elif self.backend == "mlx":
            from mlx_lm import generate
            msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + self.history
            formatted = self.llm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            response = generate(self.llm_model, self.llm_tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)

        else:  # transformers
            import torch
            msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + self.history
            formatted = self.llm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.llm_tokenizer(formatted, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                out = self.llm_model.generate(**inputs, max_new_tokens=max_tokens)
            response = self.llm_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        self.history.append({"role": "assistant", "content": response})
        self._trim_history()
        return response

    def speak(self, text: str, voice: str = "af_heart"):
        output_path = Path(__file__).parent / "response.wav"
        try:
            chunks = [audio for _, _, audio in self.tts(text[:TTS_MAX_CHARS], voice=voice)]
            if not chunks:
                return
            sf.write(str(output_path), np.concatenate(chunks), 24000)
        except Exception as e:
            print(f"  TTS error: {e}")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", str(output_path)], check=True)
            else:
                subprocess.run(["aplay", str(output_path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Audio playback error: {e}")

    def run(self):
        print("💬 Voice Chatbot — type 'quit' to exit\n")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                print("👋 Goodbye!")
                break

            response = self.chat(user_input)
            print(f"Bot: {response}\n")
            self.speak(response)

if __name__ == "__main__":
    model_id = select_model()
    personality = select_personality()
    bot = VoiceChatBot(model_id=model_id, system_prompt=personality)
    bot.run()
