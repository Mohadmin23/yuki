import sys
import io
import logging
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
import uvicorn

sys.path.append(str(Path(__file__).parent.parent))
from ms_llama import VoiceChatBot, TTS_MAX_CHARS, MAX_INPUT_CHARS, select_model, select_personality

logger = logging.getLogger(__name__)

bot: VoiceChatBot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    model_id = select_model()
    personality = select_personality()
    bot = VoiceChatBot(model_id=model_id, system_prompt=personality)
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v[:MAX_INPUT_CHARS]

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text()

@app.post("/chat")
async def chat(req: ChatRequest):
    if bot is None:
        return JSONResponse({"error": "Model is still loading, please wait."}, status_code=503)
    try:
        import asyncio
        response = await asyncio.to_thread(bot.chat, req.message)

        chunks = await asyncio.to_thread(lambda: [audio for _, _, audio in bot.tts(response[:TTS_MAX_CHARS], voice="af_heart")])
        if not chunks:
            return JSONResponse({"error": "TTS returned no audio."}, status_code=500)

        audio_data = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, audio_data, 24000, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()

        return JSONResponse({"text": response, "audio": audio_b64})
    except Exception as e:
        logger.exception("Error in /chat")
        return JSONResponse({"error": "An internal error occurred."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
