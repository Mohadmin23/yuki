from .yuki_delete import tool_yuki_delete as tool_fn

META = {
    "cli_name": "/yuki-delete",
    "react_name": "yuki_delete",
    "help": "Delete a file from yuki/",
    "usage": "/yuki-delete <filename>",
    "description": 'yuki_delete("filename") — Delete a file from yuki/ when the user asks to delete/remove it. Actually delete — do NOT pretend.',
    "param_name": "filename",
    "param_description": "Filename inside the yuki/ folder to delete. Plain name, no path.",
    "is_factual": False,
}
