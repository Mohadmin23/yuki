import re
from .image import tool_image as tool_fn

META = {
    "cli_name": "/image",
    "react_name": "image",
    "help": "Generate an image from a text prompt",
    "usage": "/image <prompt>",
    "description": 'image("prompt") — ONLY when the user EXPLICITLY asks for a picture, drawing, image, artwork, illustration, or says "draw/sketch/paint/generate/make me an image of X". NEVER call this for text tasks. The prompt must be a detailed visual description, not the user\'s raw words. Call at most ONCE per request.',
    "param_name": "prompt",
    "param_description": "Detailed visual description of the image to generate. Be specific about subject, style, colors, mood — not the user's raw words.",
    "is_factual": False,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:generate|create|draw|make|paint)\s+(?:an?\s+)?"
                r"(?:image|picture|illustration|drawing|photo)\s+"
                r"(?:of\s+|based\s+on[:\s]+|about\s+|for\s+)?(.+?)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
        {
            "pattern": re.compile(
                r"\b(?:generate|create|draw|make|paint)\s+(?:me\s+)?(.+?)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
    ],
}
