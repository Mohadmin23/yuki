from .yuki_read import tool_yuki_read as tool_fn

META = {
    "cli_name": "/yuki-read",
    "react_name": "yuki_read",
    "help": "Read a file from yuki/",
    "usage": "/yuki-read <filename>",
    "description": 'yuki_read("filename") — Read a file from yuki/ when the user asks what\'s in it.',
    "param_name": "filename",
    "param_description": "Filename inside the yuki/ folder to read. Plain name, no path.",
    "is_factual": False,
}
