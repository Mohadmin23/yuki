import re
from .hardware import tool_hardware as tool_fn

META = {
    "cli_name": "/hardware",
    "react_name": "hardware",
    "help": "Show hardware usage (cpu/ram/disk/process/top/gpu/temp/all)",
    "usage": "/hardware [metric]",
    "description": 'hardware("metric") — Only when the user asks about CPU/RAM/disk/temperature. metric = cpu, ram, disk, process, top, gpu, temp, or all.',
    "param_name": "metric",
    "param_description": "Which hardware stat to report. One of: cpu, ram, disk, process, top, gpu, temp, all. Defaults to 'all' if omitted.",
    "is_factual": True,
    # Order matters — specific patterns must come before generic ram/cpu fallbacks.
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:"
                r"(?:what|which|the)\s+(?:program|process|app|one)s?\b.*?\b(?:using|causing|hogging|eating)"
                r"|\bis\s+(?:using|causing|hogging|eating)\s+(?:so\s+much|most|all|the|it|up|my)"
                r"|top\s+(?:process|processes|procs)"
                r")",
                re.IGNORECASE,
            ),
            "arg": "top",
        },
        {
            "pattern": re.compile(
                r"\b(?:"
                r"how\s+much\s+(?:ram|memory|cpu)?\s*(?:are\s+you|do\s+you|you\s+are|you['’]?re)\s+(?:using|use|eating|hogging)"
                r"|what(?:'?s|\s+is)\s+your\s+(?:cpu|ram|memory|usage|footprint)"
                r")\b",
                re.IGNORECASE,
            ),
            "arg": "process",
        },
        {
            "pattern": re.compile(
                r"\b(?:temperature|how\s+hot|cpu\s+temp|how\s+warm\s+(?:are\s+you|is\s+(?:the\s+)?(?:cpu|chip|mac)))\b",
                re.IGNORECASE,
            ),
            "arg": "temp",
        },
        {
            "pattern": re.compile(
                r"\b(?:how\s+much\s+(?:ram|memory)|memory\s+usage|ram\s+usage)\b",
                re.IGNORECASE,
            ),
            "arg": "ram",
        },
        {
            "pattern": re.compile(
                r"\b(?:cpu\s+usage|how\s+much\s+cpu|processor\s+usage)\b", re.IGNORECASE
            ),
            "arg": "cpu",
        },
        {
            "pattern": re.compile(
                r"\b(?:hardware\s+(?:usage|stats|status)|system\s+(?:usage|stats|status)|resource\s+usage)\b",
                re.IGNORECASE,
            ),
            "arg": "all",
        },
    ],
}
