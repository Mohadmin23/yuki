from .recall import tool_recall as tool_fn

META = {
    "cli_name": "/recall",
    "react_name": "recall",
    "help": "Retrieve past conversations and Yuki notes by topic",
    "usage": "/recall <topic>",
    "description": (
        'recall("topic") — Retrieve past conversations and Yuki-owned notes related to a topic. '
        "Call this when the user references something from before that you don't see in the "
        'current chat: "remember when we...", "last time we talked about...", "that thing I '
        'told you about my X", "did I ever mention...", or any question about past events/'
        "projects/feelings that aren't in this conversation's history. The query should be a "
        'few words capturing the topic ("server crash debugging", "my sister\'s wedding"), not '
        "the user's full sentence. Use the returned summaries as fuzzy memory to inform your "
        "reply — don't quote them verbatim or list them as bullets. Use yuki_read instead when "
        "the user names a specific Yuki file and asks for its exact contents. If nothing comes "
        "back, say so honestly instead of making something up."
    ),
    "param_name": "topic",
    "param_description": "A few words capturing the topic to retrieve from old conversations or Yuki notes. Not a filename and not the user's full sentence.",
    "is_factual": False,
}
