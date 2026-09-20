# Production cleanup

The application now keeps three TTS paths:

- Chatterbox Multilingual V3 — primary German/multilingual audiobook engine
- Kokoro — lightweight English engine
- Qwen3-TTS — optional fallback

The proven Chatterbox Fast Quality CUDA Graph backend is retained. Temporary profiling, FP16, persistent-graph, EOS-batching, S3 CUDA-graph, parameter-cache and torch.compile experiments are not part of the production code.

`Finalize-Code-Cleanup.bat` only removes stale development files. It does not touch `.audiobook_data`, installed Chatterbox/Kokoro/Qwen models, or user projects.
