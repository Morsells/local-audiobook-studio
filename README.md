# Local Audiobook Studio

Privacy-first Windows application for turning locally owned PDF/EPUB/TXT/Markdown books into narrated audiobooks.

## Retained production stack

- **Chatterbox Multilingual V3** — recommended German/multilingual narration; FP32 Fast Quality uses the validated T3 CUDA Graph path when CUDA is available.
- **Kokoro** — lightweight local English narration.
- **Qwen3-TTS** — optional local multilingual fallback.
- **Ollama / TranslateGemma** plus local translation alternatives for EN↔DE workflows.
- Chapter-aware M4B export, queue/resume, pronunciation/replacement rules, persistent local project data and offline runtime guards.

Rejected TTS backends and temporary optimization experiments have been removed from the production codebase.

## Start

```powershell
.\Start-Audiobook-Studio-With-Chatterbox.bat
```

Normal automatic launcher:

```powershell
.\Start-Audiobook-Studio.bat
```

Translation-only mode:

```powershell
.\Start-Audiobook-Studio-Translate.bat
```

Stop/status:

```powershell
.\Stop-Audiobook-Studio.bat
.\Status-Audiobook-Studio.bat
```

## Local data

`.audiobook_data/` contains project state, translations, audio caches, output and project-local Chatterbox reference clips. Do not delete it when updating the application.

Normal model runtime is local-only. Chatterbox loads model assets through `from_local()` and the sidecars bind to loopback addresses.
