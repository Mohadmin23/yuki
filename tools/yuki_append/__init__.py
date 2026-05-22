from .yuki_append import tool_yuki_append as tool_fn

META = {
    "cli_name": "/yuki-append",
    "react_name": "yuki_append",
    "help": "Append to a file in yuki/",
    "usage": "/yuki-append <filename>|<content>",
    "description": 'yuki_append("filename|content") — Append to an existing yuki/ file when the user asks to add to it.',
    "param_name": "filename_and_content",
    "param_description": "Pipe-separated 'filename|content' to append into a yuki/ file that already exists.",
    "is_factual": False,
}
