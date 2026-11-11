import re

_STRICT = re.compile(r"(?im)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)")
_LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")

def extract_answer(text: str, mode: str) -> str | None:
    match = _STRICT.search(text) or _LOOSE.search(text)
    if not match:
        return None
    result = match.group(1).strip().rstrip(".")
    return result.upper() if mode == "multiple_choice" else result.lower()
