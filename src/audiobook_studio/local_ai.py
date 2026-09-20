from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from .privacy import assert_local_url

OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_available(timeout: float = 1.0) -> bool:
    assert_local_url(OLLAMA_URL)
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def list_models(timeout: float = 2.0) -> list[str]:
    assert_local_url(OLLAMA_URL)
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [
            model.get("name", "")
            for model in data.get("models", [])
            if model.get("name")
        ]
    except Exception:
        return []


def _generate(prompt: str, model: str, timeout: float = 900.0) -> str:
    assert_local_url(OLLAMA_URL)
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", "")).strip()
    except URLError as exc:
        raise RuntimeError(
            "Could not reach local Ollama. Start Ollama first."
        ) from exc


def propose_spoken_rewrite(
    text: str,
    model: str,
    timeout: float = 900.0,
) -> str:
    prompt = (
        "Rewrite the following book passage only for spoken narration. "
        "Preserve every substantive claim, name, quotation meaning, and factual detail. "
        "Remove only formatting artifacts, citations, awkward cross-references, and notation "
        "that is unpleasant to hear. Do not summarize and do not add facts. "
        "Return only the proposed narration text.\n\nPASSAGE:\n"
        + text
    )
    return _generate(prompt, model, timeout)


def review_translation(
    source_text: str,
    translated_text: str,
    model: str,
    *,
    source_language: str = "English",
    target_language: str = "German",
) -> str:
    prompt = f"""Act as a careful book translation editor.

Compare the SOURCE with the EXISTING TRANSLATION.
Improve the translation only where necessary.

Requirements:
- Preserve every substantive claim and factual detail.
- Do not summarize or add commentary.
- Preserve names and quotation meaning.
- Produce fluent, idiomatic {target_language}.
- Fix mistranslations, awkward literal phrasing, and inconsistent terminology.
- Return only the complete corrected {target_language} translation.

SOURCE ({source_language}):
{source_text}

EXISTING TRANSLATION ({target_language}):
{translated_text}
"""
    return _generate(prompt, model, 900.0)
