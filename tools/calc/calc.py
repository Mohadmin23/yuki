"""Safe math expression evaluator."""
from __future__ import annotations


def tool_calc(expression):
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.()% ")
    if not all(c in allowed for c in expression):
        return "Invalid expression. Only numbers and +-*/.()% allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"
