from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import TranslationEngine
from .hardware import configure_torch_cpu_threads, detect_hardware, resolve_device, resolve_precision
from .privacy import assert_local_url


LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}


OPUS_MODELS = {
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
}


def stable_hash(value) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def translation_model_dir(models_root: Path, source: str, target: str) -> Path:
    return models_root / "translation" / f"opus-mt-{source}-{target}"


def _mask_glossary(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    masked = text
    placeholders: dict[str, str] = {}
    for index, source_term in enumerate(sorted(glossary, key=len, reverse=True)):
        target_term = str(glossary[source_term]).strip()
        if not source_term.strip() or not target_term:
            continue
        marker = f"[GLOSS{index:03d}]"
        pattern = re.compile(rf"(?<!\w){re.escape(source_term)}(?!\w)", re.IGNORECASE)
        if pattern.search(masked):
            masked = pattern.sub(marker, masked)
            placeholders[marker] = target_term
    return masked, placeholders


def _restore_glossary(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for marker, target in placeholders.items():
        variants = {
            marker,
            marker.lower(),
            marker.replace("[", "").replace("]", ""),
            marker.replace("[", "<").replace("]", ">"),
        }
        for variant in variants:
            restored = restored.replace(variant, target)
    return restored


def _paragraph_chunks(text: str, max_chars: int = 850) -> list[str]:
    """Create conservative prose chunks before tokenizer-aware splitting."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", paragraph)
            if part.strip()
        ] or [paragraph]

        current = ""
        for sentence in sentences:
            sentence_parts = [sentence]
            if len(sentence) > max_chars:
                sentence_parts = [
                    part.strip()
                    for part in re.split(r"(?<=[;:])\s+|(?<=,)\s+", sentence)
                    if part.strip()
                ] or [sentence]

            for part in sentence_parts:
                proposed = (current + " " + part).strip()
                if current and len(proposed) > max_chars:
                    chunks.append(current)
                    current = part
                else:
                    current = proposed

                if len(current) > max_chars:
                    chunks.append(current)
                    current = ""

        if current:
            chunks.append(current)

    return chunks


class OpusTranslator:
    def __init__(
        self,
        model_dir: Path,
        *,
        hardware_mode: str = "auto",
        batch_size: int = 0,
    ):
        if not model_dir.exists():
            raise RuntimeError(
                f"Local OPUS model is missing: {model_dir}\n"
                "Run `python scripts/download_translation_models.py --pair en-de` first."
            )
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OPUS translation dependencies are missing. "
                "Run `pip install -r requirements-translation.txt`."
            ) from exc

        self.torch = torch
        self.hardware = detect_hardware()
        self.device = resolve_device(hardware_mode, self.hardware)
        self.precision = resolve_precision(hardware_mode, self.hardware)

        auto_batch = int(self.hardware.recommended_translation_batch or 1)
        self.batch_size = max(1, int(batch_size or auto_batch))

        if self.device == "cpu":
            configure_torch_cpu_threads(torch)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )

        # Marian/OPUS models are substantially faster in FP16 on NVIDIA GPUs.
        # BF16 is selected only when Torch explicitly reports hardware support.
        if self.device == "cuda":
            if self.precision == "bfloat16":
                self.model = self.model.to(dtype=torch.bfloat16)
            elif self.precision == "float16":
                self.model = self.model.to(dtype=torch.float16)

        self.model.to(self.device)
        self.model.eval()

        # Marian/OPUS uses static positional embeddings. Keep source and decoder
        # generation comfortably below the model's declared hard limit.
        position_limit = int(
            getattr(self.model.config, "max_position_embeddings", 512) or 512
        )
        self.encoder_position_limit = position_limit
        self.max_source_tokens = max(96, min(320, position_limit - 64))
        self.max_target_tokens = max(96, min(480, position_limit - 16))

        input_embeddings = self.model.get_input_embeddings()
        self.input_vocab_size = int(
            getattr(input_embeddings, "num_embeddings", 0)
            or getattr(self.model.config, "vocab_size", 0)
        )

    @property
    def backend_label(self) -> str:
        return (
            f"{self.device} · {self.precision} · batch {self.batch_size}"
        )

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        ids = encoded.get("input_ids", [])
        return len(ids)

    def _split_token_safe(self, text: str) -> list[str]:
        """Split prose without silently dropping source tokens."""
        text = text.strip()
        if not text:
            return []

        if self._token_count(text) <= self.max_source_tokens:
            return [text]

        candidates = [
            part.strip()
            for part in re.split(
                r"(?<=[.!?])\s+|(?<=[;:])\s+|(?<=,)\s+",
                text,
            )
            if part.strip()
        ]

        if len(candidates) <= 1:
            words = text.split()
            pieces: list[str] = []
            current = ""
            for word in words:
                proposed = (current + " " + word).strip()
                if current and self._token_count(proposed) > self.max_source_tokens:
                    pieces.append(current)
                    current = word
                else:
                    current = proposed
            if current:
                pieces.append(current)
            return pieces

        pieces: list[str] = []
        current = ""
        for candidate in candidates:
            proposed = (current + " " + candidate).strip()
            if current and self._token_count(proposed) > self.max_source_tokens:
                pieces.extend(self._split_token_safe(current))
                current = candidate
            else:
                current = proposed

        if current:
            if self._token_count(current) > self.max_source_tokens:
                pieces.extend(self._split_token_safe(current))
            else:
                pieces.append(current)

        return pieces

    def _safe_chunks(self, text: str) -> list[str]:
        safe: list[str] = []
        for chunk in _paragraph_chunks(text):
            safe.extend(self._split_token_safe(chunk))
        return [chunk for chunk in safe if chunk.strip()]

    def _translate_batch(
        self,
        chunks: list[str],
        glossary: dict[str, str],
    ) -> list[str]:
        masked_chunks: list[str] = []
        placeholder_sets: list[dict[str, str]] = []

        for chunk in chunks:
            masked, placeholders = _mask_glossary(chunk, glossary)
            masked_chunks.append(masked)
            placeholder_sets.append(placeholders)

        inputs = self.tokenizer(
            masked_chunks,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            sequence_length = int(input_ids.shape[-1])
            if sequence_length > self.encoder_position_limit:
                raise RuntimeError(
                    "OPUS source chunk exceeded the model's positional limit "
                    f"({sequence_length} > {self.encoder_position_limit}). "
                    "This should have been split automatically."
                )

            if self.input_vocab_size > 0 and input_ids.numel():
                min_id = int(input_ids.min().item())
                max_id = int(input_ids.max().item())
                if min_id < 0 or max_id >= self.input_vocab_size:
                    raise RuntimeError(
                        "OPUS tokenizer/model vocabulary mismatch detected before CUDA "
                        f"(token id range {min_id}..{max_id}, "
                        f"embedding size {self.input_vocab_size})."
                    )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_target_tokens,
                    num_beams=4,
                    early_stopping=True,
                )
        except RuntimeError as exc:
            if "device-side assert" in str(exc).lower():
                raise RuntimeError(
                    "OPUS CUDA hit a device-side assert. The current CUDA context "
                    "cannot be reused safely. Restart Local Audiobook Studio; all "
                    "translations saved before this section remain cached."
                ) from exc
            raise

        translated_batch = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        return [
            _restore_glossary(translated, placeholders)
            for translated, placeholders in zip(
                translated_batch,
                placeholder_sets,
                strict=False,
            )
        ]

    def translate(
        self,
        text: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        glossary = glossary or {}
        chunks = self._safe_chunks(text)
        output: list[str] = []

        for index in range(0, len(chunks), self.batch_size):
            batch = chunks[index:index + self.batch_size]
            output.extend(self._translate_batch(batch, glossary))

        return "\n\n".join(output).strip()


class ArgosTranslator:
    def __init__(self, source: str, target: str):
        try:
            import argostranslate.translate
        except ImportError as exc:
            raise RuntimeError(
                "Argos Translate is optional and is not installed. "
                "Install `argostranslate`, then install the desired local language package."
            ) from exc
        self.translate_module = argostranslate.translate
        self.source = source
        self.target = target

    def translate(self, text: str, glossary: dict[str, str] | None = None) -> str:
        masked, placeholders = _mask_glossary(text, glossary or {})
        translated = self.translate_module.translate(masked, self.source, self.target)
        return _restore_glossary(translated, placeholders)


_LITERARY_SUMMARY_MARKERS = (
    "here's a breakdown", "here is a breakdown", "core themes", "key themes",
    "key ideas and concepts", "key ideas", "this is a fascinating", "this excerpt",
    "the excerpt", "the text discusses", "the text explores", "the text emphasizes",
    "summary:", "in summary", "kernthemen", "schlüsselideen", "zusammenfassung:",
    "dieser auszug", "der text behandelt",
)

_ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "that", "is", "for", "with", "as",
    "this", "it", "you", "your", "are", "be", "on", "from", "by", "or",
    "an", "a", "was", "were", "have", "has", "but", "not", "their", "they",
}

_GERMAN_FUNCTION_WORDS = {
    "der", "die", "das", "und", "zu", "in", "den", "von", "mit", "dass",
    "ist", "für", "sich", "auf", "ein", "eine", "als", "auch", "nicht",
    "sie", "du", "dein", "deine", "werden", "wird", "dem", "des", "im",
    "zum", "zur", "ihnen", "ihre", "aber", "wie", "wenn", "oder", "er",
}


def _word_tokens(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+)?",
        text,
    )


def _split_literary_translation_chunks(
    text: str,
    *,
    max_words: int = 600,
    max_chars: int = 4200,
) -> list[str]:
    """Split long book sections conservatively while preserving paragraph order."""
    text = text.strip()
    if not text:
        return []

    def count_words(value: str) -> int:
        return len(_word_tokens(value))

    def split_oversized_paragraph(paragraph: str) -> list[str]:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", paragraph)
            if part.strip()
        ] or [paragraph.strip()]
        pieces: list[str] = []
        current: list[str] = []
        current_words = 0
        current_chars = 0

        def flush() -> None:
            nonlocal current, current_words, current_chars
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_words = 0
                current_chars = 0

        for sentence in sentences:
            sentence_words = count_words(sentence)
            if sentence_words > max_words or len(sentence) > max_chars:
                flush()
                words = sentence.split()
                group: list[str] = []
                group_chars = 0
                for word in words:
                    projected_chars = group_chars + (1 if group else 0) + len(word)
                    if group and (len(group) >= max_words or projected_chars > max_chars):
                        pieces.append(" ".join(group))
                        group = []
                        group_chars = 0
                    group.append(word)
                    group_chars += (1 if group_chars else 0) + len(word)
                if group:
                    pieces.append(" ".join(group))
                continue

            projected_words = current_words + sentence_words
            projected_chars = current_chars + (1 if current else 0) + len(sentence)
            if current and (projected_words > max_words or projected_chars > max_chars):
                flush()
            current.append(sentence)
            current_words += sentence_words
            current_chars += (1 if current_chars else 0) + len(sentence)

        flush()
        return pieces

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    current_chars = 0

    def flush_current() -> None:
        nonlocal current, current_words, current_chars
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0
            current_chars = 0

    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)
        if paragraph_words > max_words or len(paragraph) > max_chars:
            flush_current()
            chunks.extend(split_oversized_paragraph(paragraph))
            continue

        projected_words = current_words + paragraph_words
        projected_chars = current_chars + (2 if current else 0) + len(paragraph)
        if current and (projected_words > max_words or projected_chars > max_chars):
            flush_current()
        current.append(paragraph)
        current_words += paragraph_words
        current_chars += (2 if current_chars else 0) + len(paragraph)

    flush_current()
    return chunks


def _translation_quality_issues(
    source: str,
    translated: str,
    *,
    target_language: str,
) -> list[str]:
    """Catch obvious summary/language/truncation failures before caching."""
    issues: list[str] = []
    source = source.strip()
    translated = translated.strip()
    if not translated:
        return ["empty output"]

    source_words = _word_tokens(source)
    output_words = _word_tokens(translated)
    source_count = len(source_words)
    output_count = len(output_words)

    if source_count >= 40:
        ratio = output_count / max(1, source_count)
        if ratio < 0.62:
            issues.append(f"output is suspiciously short ({output_count}/{source_count} words)")
        elif ratio > 1.80:
            issues.append(f"output is suspiciously long ({output_count}/{source_count} words)")

    source_lower = source.lower()
    output_lower = translated.lower()
    for marker in _LITERARY_SUMMARY_MARKERS:
        if marker in output_lower and marker not in source_lower:
            issues.append(f"added summary/commentary marker: {marker!r}")
            break

    if target_language == "de" and output_count >= 12:
        lower_words = [word.lower() for word in output_words]
        english_hits = sum(word in _ENGLISH_FUNCTION_WORDS for word in lower_words)
        german_hits = sum(word in _GERMAN_FUNCTION_WORDS for word in lower_words)
        if english_hits >= 6 and english_hits > max(5, int(german_hits * 1.35)):
            issues.append(
                f"output appears predominantly English ({english_hits} EN vs {german_hits} DE function words)"
            )

    source_paragraphs = [p for p in re.split(r"\n\s*\n", source) if p.strip()]
    output_paragraphs = [p for p in re.split(r"\n\s*\n", translated) if p.strip()]
    if source_count >= 120 and len(source_paragraphs) >= 4:
        minimum = max(2, (len(source_paragraphs) + 1) // 2)
        if len(output_paragraphs) < minimum:
            issues.append(
                f"paragraph structure collapsed ({len(output_paragraphs)}/{len(source_paragraphs)} paragraphs)"
            )

    return issues


class OllamaTranslator:
    def __init__(
        self,
        model: str,
        url: str = "http://127.0.0.1:11434",
        *,
        german_address_style: str = "du",
    ):
        if not model.strip():
            raise RuntimeError("Choose a local Ollama model first.")
        assert_local_url(url)
        self.model = model
        self.url = url.rstrip("/")
        self.german_address_style = (
            german_address_style if german_address_style in {"du", "Sie"} else "du"
        )

    @property
    def is_translate_gemma(self) -> bool:
        return "translategemma" in self.model.lower()

    def _translate_gemma_prompt(
        self,
        text: str,
        glossary: dict[str, str],
        *,
        source_language: str,
        target_language: str,
        segment_index: int = 1,
        segment_total: int = 1,
        retry_issues: list[str] | None = None,
    ) -> str:
        source_name = LANGUAGE_NAMES.get(source_language, source_language)
        target_name = LANGUAGE_NAMES.get(target_language, target_language)
        glossary_text = "\n".join(
            f"- {src} => {dst}" for src, dst in glossary.items()
        )

        extra_rules: list[str] = [
            "Translate the ENTIRE CURRENT PASSAGE. This is a translation task, not an analysis task.",
            "Translate every sentence and paragraph in order; do not summarize or compress the passage.",
            "Do not analyze, explain, introduce, review, or comment on the text.",
            "Do not add headings, bullet lists, themes, takeaways, notes, or conclusions that are absent from the source.",
            "Preserve every substantive claim, name, historical detail, quotation meaning, and rhetorical emphasis.",
            "Do not omit, soften, censor, or add content.",
            "Use idiomatic target-language sentence structure rather than copying source-language word order.",
            "Preserve paragraph boundaries and meaningful headings.",
            f"Return {target_name} translation only; do not answer in {source_name} except where a proper noun or quoted wording genuinely requires it.",
        ]
        if target_language == "de":
            extra_rules.extend(
                [
                    "Write natural modern Standard German (Hochdeutsch) suitable for fluent audiobook narration.",
                    "Avoid English calques, literal English syntax, stiff machine-translation phrasing, and unnecessary Anglicisms.",
                    "Keep the author's authoritative, strategic, literary tone without making it archaic.",
                    (
                        "For generic second-person advice, address the listener consistently with informal singular 'du'."
                        if self.german_address_style == "du"
                        else "For generic second-person advice, address the listener consistently with formal 'Sie'."
                    ),
                ]
            )
        if glossary_text:
            extra_rules.append(
                "Use these terminology mappings whenever the source term occurs:\n"
                + glossary_text
            )
        if retry_issues:
            extra_rules.append(
                "A previous attempt was rejected by an automatic fidelity guard for: "
                + "; ".join(retry_issues)
                + ". Correct those failures completely."
            )

        rules = "\n".join(f"- {rule}" for rule in extra_rules)
        segment_note = (
            f"This is source segment {segment_index} of {segment_total}. "
            "Translate only this CURRENT PASSAGE; never invent missing surrounding text."
            if segment_total > 1
            else "Translate only the CURRENT PASSAGE below."
        )
        return (
            f"You are a professional {source_name} ({source_language}) to "
            f"{target_name} ({target_language}) book translator.\n"
            f"{segment_note}\n\n"
            f"NON-NEGOTIABLE REQUIREMENTS:\n{rules}\n\n"
            f"CURRENT PASSAGE:\n{text}"
        )

    def _literary_prompt(
        self,
        text: str,
        glossary: dict[str, str],
        *,
        source_language: str,
        target_language: str,
        previous_context: str,
    ) -> str:
        glossary_text = "\n".join(f"- {src} => {dst}" for src, dst in glossary.items())
        context = previous_context[-2500:].strip()
        source_name = LANGUAGE_NAMES.get(source_language, source_language)
        target_name = LANGUAGE_NAMES.get(target_language, target_language)
        german_style = ""
        if target_language == "de":
            address_rule = (
                "Use informal singular 'du' consistently for generic second-person advice."
                if self.german_address_style == "du"
                else "Use formal 'Sie' consistently for generic second-person advice."
            )
            german_style = f"""
German audiobook style:
- Write natural modern Standard German (Hochdeutsch), not translated-sounding German.
- Restructure sentences freely when necessary to avoid English word order or calques.
- Prefer idiomatic German verbs, collocations, and punctuation.
- Keep the author's authoritative, strategic, literary tone without becoming archaic.
- {address_rule}
"""

        return f"""Translate the CURRENT PASSAGE from {source_name} to {target_name} as publication-quality book prose.

Non-negotiable fidelity rules:
- Preserve every substantive claim, fact, name, historical detail, quotation meaning, and rhetorical emphasis.
- Do not summarize, omit, soften, censor, explain, or add content.
- Preserve meaningful headings and paragraph boundaries.
- Sentence structure may change whenever that is necessary for idiomatic {target_name}.
- Use the glossary consistently when a listed term occurs.
- Return only the translated CURRENT PASSAGE, with no preface, notes, or commentary.
- The previous passage is context only; never translate or repeat it.
{german_style}
GLOSSARY:
{glossary_text or "(none)"}

PREVIOUS PASSAGE FOR CONTEXT:
{context or "(none)"}

CURRENT PASSAGE:
{text}
"""

    def _request_ollama(
        self,
        prompt: str,
        *,
        translate_gemma: bool,
        strict_retry: bool = False,
    ) -> str:
        if translate_gemma:
            endpoint = f"{self.url}/api/chat"
            payload_dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.10 if strict_retry else 0.20,
                    "top_p": 0.90,
                    "top_k": 64,
                    "num_ctx": 4096,
                    "num_predict": 2048,
                },
            }
        else:
            endpoint = f"{self.url}/api/generate"
            payload_dict = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "num_predict": 2048,
                },
            }

        payload = json.dumps(payload_dict).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=1800) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw_detail = ""
            try:
                raw_detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                pass

            detail = raw_detail
            if raw_detail:
                try:
                    parsed = json.loads(raw_detail)
                    if isinstance(parsed, dict):
                        detail = str(parsed.get("error") or parsed.get("message") or raw_detail)
                except Exception:
                    pass

            lower_detail = detail.lower()
            hint = ""
            if any(token in lower_detail for token in (
                "out of memory", "memory", "cuda", "runner process",
                "llama-server", "unable to load model",
            )):
                hint = (
                    " The Ollama HTTP server is reachable, but the model/runtime failed. "
                    "Run `ollama run translategemma:12b` in PowerShell to verify the "
                    "model independently of the studio."
                )

            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for {self.model!r}: "
                f"{detail or exc.reason}.{hint}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to local Ollama at 127.0.0.1:11434. "
                "Start Ollama and try again."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Local Ollama request failed for {self.model!r}: {exc}"
            ) from exc

        if translate_gemma:
            message = data.get("message") or {}
            result = str(message.get("content", "")).strip()
        else:
            result = str(data.get("response", "")).strip()

        if not result:
            raise RuntimeError(
                f"Local Ollama model {self.model!r} returned an empty translation."
            )
        return result

    def _translate_gemma_chunk(
        self,
        text: str,
        glossary: dict[str, str],
        *,
        source_language: str,
        target_language: str,
        segment_index: int,
        segment_total: int,
        depth: int = 0,
    ) -> str:
        last_result = ""
        last_issues: list[str] = []

        for attempt in range(2):
            prompt = self._translate_gemma_prompt(
                text,
                glossary,
                source_language=source_language,
                target_language=target_language,
                segment_index=segment_index,
                segment_total=segment_total,
                retry_issues=last_issues if attempt else None,
            )
            result = self._request_ollama(
                prompt,
                translate_gemma=True,
                strict_retry=attempt > 0,
            )
            issues = _translation_quality_issues(
                text,
                result,
                target_language=target_language,
            )
            if not issues:
                return result
            last_result = result
            last_issues = issues

        # If a passage still drifts, make it materially smaller and retry rather
        # than accepting a summary/English answer into the persistent cache.
        source_word_count = len(_word_tokens(text))
        if depth < 2 and source_word_count >= 140:
            fallback_max_words = max(90, min(300, source_word_count // 2))
            smaller = _split_literary_translation_chunks(
                text,
                max_words=fallback_max_words,
                max_chars=2400,
            )
            if len(smaller) > 1:
                translated_parts: list[str] = []
                for sub_index, part in enumerate(smaller, start=1):
                    translated_parts.append(
                        self._translate_gemma_chunk(
                            part,
                            glossary,
                            source_language=source_language,
                            target_language=target_language,
                            segment_index=sub_index,
                            segment_total=len(smaller),
                            depth=depth + 1,
                        )
                    )
                return "\n\n".join(translated_parts).strip()

        preview = re.sub(r"\s+", " ", last_result)[:220]
        raise RuntimeError(
            "TranslateGemma output was rejected by the fidelity guard after retries: "
            + "; ".join(last_issues)
            + (f". Output preview: {preview!r}" if preview else "")
        )

    def translate(
        self,
        text: str,
        glossary: dict[str, str] | None = None,
        *,
        source_language: str,
        target_language: str,
        previous_context: str = "",
    ) -> str:
        glossary = glossary or {}

        if self.is_translate_gemma:
            chunks = _split_literary_translation_chunks(text)
            if not chunks:
                return ""

            translated_chunks: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                translated_chunks.append(
                    self._translate_gemma_chunk(
                        chunk,
                        glossary,
                        source_language=source_language,
                        target_language=target_language,
                        segment_index=index,
                        segment_total=len(chunks),
                    )
                )
            result = "\n\n".join(translated_chunks).strip()

            final_issues = _translation_quality_issues(
                text,
                result,
                target_language=target_language,
            )
            severe = [
                issue for issue in final_issues
                if not issue.startswith("paragraph structure collapsed")
            ]
            if severe:
                raise RuntimeError(
                    "TranslateGemma assembled section failed the fidelity guard: "
                    + "; ".join(severe)
                )
            return result

        prompt = self._literary_prompt(
            text,
            glossary,
            source_language=source_language,
            target_language=target_language,
            previous_context=previous_context,
        )
        return self._request_ollama(prompt, translate_gemma=False)


@dataclass
class TranslationRecord:
    section_id: str
    source_hash: str
    glossary_hash: str
    source_language: str
    target_language: str
    engine: str
    model: str
    translated_text: str
    manual_text: str = ""
    updated_at: str = ""

    @property
    def effective_text(self) -> str:
        return self.manual_text.strip() or self.translated_text.strip()

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "source_hash": self.source_hash,
            "glossary_hash": self.glossary_hash,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "engine": self.engine,
            "model": self.model,
            "translated_text": self.translated_text,
            "manual_text": self.manual_text,
            "updated_at": self.updated_at or datetime.now(timezone.utc).isoformat(),
        }


def translation_is_current(
    record: dict | None,
    source_text: str,
    glossary: dict[str, str],
    *,
    source_language: str,
    target_language: str,
    engine: TranslationEngine,
    model: str,
) -> bool:
    if not record:
        return False
    return (
        record.get("source_hash") == stable_hash(source_text)
        and record.get("glossary_hash") == stable_hash(glossary)
        and record.get("source_language") == source_language
        and record.get("target_language") == target_language
        and record.get("engine") == engine.value
        and record.get("model", "") == model
        and bool(record.get("translated_text", "").strip())
    )
