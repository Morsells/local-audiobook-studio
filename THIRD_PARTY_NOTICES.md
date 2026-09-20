# Third-party notices

This project does not bundle third-party model weights in update ZIPs.

## Chatterbox Multilingual V3

Resemble AI Chatterbox is used for primary multilingual narration. The runtime loads local model assets through the upstream `from_local()` API. Generated Chatterbox audio includes the upstream PerTh perceptual watermark. Consult the upstream repository/model card for redistribution terms.

## Kokoro

Kokoro is used as the lightweight local English narration engine. Consult the model/package sources for current model and code licenses before redistributing weights.

## Qwen3-TTS

Qwen3-TTS is retained as an optional multilingual fallback. Official project: QwenLM/Qwen3-TTS. Consult the upstream project/model card for current redistribution terms.

## Translation models

Local translation can use Ollama-hosted models, Helsinki-NLP OPUS-MT and Argos Translate. Model licenses vary by selected model; check the corresponding upstream model card before redistributing model assets.

## Application dependencies

See installed package metadata and the respective upstream projects for current third-party license terms.
