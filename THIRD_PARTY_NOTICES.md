# Third-party notices

This repository does not track third-party model weights. Setup scripts download model assets into the Git-ignored `models/` directory.

## Chatterbox Multilingual V3

Resemble AI Chatterbox is used for primary multilingual narration.

- Upstream: https://github.com/resemble-ai/chatterbox
- The project pins the V3-capable upstream source used by the local sidecar.
- Runtime model assets are loaded locally through the upstream `from_local()` API.
- Generated Chatterbox audio includes the upstream PerTh perceptual watermark.

Review the upstream code/model licenses before redistributing Chatterbox code or weights.

## Kokoro

Kokoro is used as the lightweight local English narration engine through `kokoro-onnx`.

- Upstream runtime: https://github.com/thewh1teagle/kokoro-onnx

Review the model and runtime licenses before redistributing model files.

## Qwen3-TTS

Qwen3-TTS is retained as an optional multilingual fallback.

- Upstream: https://github.com/QwenLM/Qwen3-TTS

Review the upstream project/model card for current redistribution terms.

## Translation

Local translation can use several independently licensed components, including:

- Ollama: https://ollama.com/
- Helsinki-NLP OPUS-MT models
- Argos Translate
- user-selected Ollama models

Model licenses vary by selected model. Check the corresponding upstream model card before redistributing model assets.

## Python dependencies

Application dependencies are declared in `pyproject.toml` and the `requirements*.txt` files. Each dependency remains subject to its own license.
