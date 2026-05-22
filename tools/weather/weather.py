"""Weather via wttr.in."""
from __future__ import annotations

from urllib.request import urlopen


def tool_weather(city):
    """Get weather for a city using wttr.in."""
    try:
        url = (
            f"https://wttr.in/{city.replace(' ', '+')}"
            f"?format=%l:+%C+%t+(feels+like+%f)+humidity+%h+wind+%w"
        )
        with urlopen(url, timeout=5) as resp:
            return resp.read().decode().strip()
    except Exception as e:
        return f"Weather error: {e}"
