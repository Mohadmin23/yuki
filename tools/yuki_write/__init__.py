from .yuki_write import tool_yuki_write as tool_fn

META = {
    "cli_name": "/yuki-write",
    "react_name": "yuki_write",
    "help": "Write a file in yuki/",
    "usage": "/yuki-write <filename>|<content>",
    "description": 'yuki_write("filename|content") — Write/overwrite a file in yuki/. Use when the user asks you to write/save/create a file with content.',
    "param_name": "filename_and_content",
    "param_description": "Pipe-separated 'filename|content' to write into the yuki/ folder. Example: 'haiku.txt|raindrops fall slowly'.",
    "is_factual": False,
}
