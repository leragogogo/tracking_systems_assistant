import re


def clean_body_text(text: str) -> str:
    """
    Basic cleaning for issue bodies:
    - remove fenced code blocks ```...```
    - remove inline code `...`
    - remove URLs
    - remove HTML tags
    - drop very noisy log/stacktrace-style lines
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove fenced code blocks ``` ... ```
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove inline code `...`
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Split into lines and remove very noisy ones (stack traces, file paths, etc.)
    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()

        # Skip stacktrace-like lines
        if re.search(r"Traceback \(most recent call last\)", line_stripped):
            continue
        if re.search(r"^\s*at\s+\S+", line_stripped):  # e.g. Java/JS stack frames
            continue
        if re.search(r"\.java:\d+|\.py:\d+|\.js:\d+|\.cs:\d+", line_stripped):
            continue

        # Keep the line if not empty after stripping
        if line_stripped:
            cleaned_lines.append(line_stripped)

    text = " ".join(cleaned_lines)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_body_text(text: str) -> str:
    """
    Basic cleaning for issue bodies:
    - remove fenced code blocks ```...```
    - remove inline code `...`
    - remove URLs
    - remove HTML tags
    - drop very noisy log/stacktrace-style lines
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove fenced code blocks ``` ... ```
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove inline code `...`
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Split into lines and remove very noisy ones (stack traces, file paths, etc.)
    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()

        # Skip stacktrace-like lines
        if re.search(r"Traceback \(most recent call last\)", line_stripped):
            continue
        if re.search(r"^\s*at\s+\S+", line_stripped):  # e.g. Java/JS stack frames
            continue
        if re.search(r"\.java:\d+|\.py:\d+|\.js:\d+|\.cs:\d+", line_stripped):
            continue

        # Keep the line if not empty after stripping
        if line_stripped:
            cleaned_lines.append(line_stripped)

    text = " ".join(cleaned_lines)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

