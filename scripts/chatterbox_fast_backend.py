from __future__ import annotations

def _make_static_cache(StaticCache, config, batch_size, max_cache_len, device, dtype):
    """Support the StaticCache constructor variants used by recent Transformers."""
    attempts = (
        lambda: StaticCache(
            config=config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        ),
        lambda: StaticCache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        ),
        lambda: StaticCache(
            config,
            batch_size,
            max_cache_len,
            device,
            dtype,
        ),
    )
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    raise RuntimeError(
        "Installed Transformers StaticCache API is incompatible with the "
        "experimental CUDA-graph backend."
    ) from last


def _validate_inputs(self, text_tokens):
    import torch

    text_tokens = torch.atleast_2d(
        text_tokens
    ).to(dtype=torch.long, device=self.device)

    batch = text_tokens.size(0)
    if int((text_tokens == self.hp.start_text_token).int().sum()) < batch:
        raise AssertionError("missing start_text_token")
    if int((text_tokens == self.hp.stop_text_token).int().sum()) < batch:
        raise AssertionError("missing stop_text_token")

    return text_tokens


def _sample_next(
    self,
    hidden_states,
    generated_ids,
    *,
    cfg_weight,
    temperature,
    repetition_penalty_processor,
    min_p_warper,
    top_p_warper,
):
    import torch

    logits_step = self.speech_head(hidden_states)[:, -1, :]

    cond = logits_step[0:1, :]
    uncond = logits_step[1:2, :]
    cfg = torch.as_tensor(
        cfg_weight,
        device=cond.device,
        dtype=cond.dtype,
    )
    logits = cond + cfg * (cond - uncond)

    ids_for_proc = generated_ids[:1, ...]
    logits = repetition_penalty_processor(ids_for_proc, logits)

    if temperature != 1.0:
        logits = logits / temperature

    logits = min_p_warper(ids_for_proc, logits)
    logits = top_p_warper(ids_for_proc, logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def inference_cuda_graph(
    self,
    *,
    t3_cond,
    text_tokens,
    initial_speech_tokens=None,
    prepend_prompt_speech_tokens=None,
    num_return_sequences=1,
    max_new_tokens=None,
    stop_on_eos=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    length_penalty=1.0,
    repetition_penalty=1.2,
    cfg_weight=0.5,
):
    """
    Production CUDA-graph decode for the pinned multilingual V3 path.

    Sampling, CFG and token-processing semantics intentionally mirror the
    official multilingual inference. Only the one-token transformer decode
    step is graph-captured.
    """
    import torch
    from transformers import StaticCache
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
        raise RuntimeError("CUDA-graph backend requires an NVIDIA CUDA device.")
    if prepend_prompt_speech_tokens is not None:
        raise AssertionError("prepend_prompt_speech_tokens is not implemented")
    if num_return_sequences != 1:
        raise ValueError("CUDA-graph backend supports num_return_sequences=1 only.")
    if not do_sample:
        raise ValueError("CUDA-graph backend currently mirrors the sampling path only.")

    text_tokens = _validate_inputs(self, text_tokens)
    max_new_tokens = max_new_tokens or self.hp.max_speech_tokens

    if initial_speech_tokens is None:
        initial_speech_tokens = (
            self.hp.start_speech_token
            * torch.ones_like(text_tokens[:, :1])
        )

    embeds, _ = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
        cfg_weight=cfg_weight,
    )

    device = embeds.device
    dtype = embeds.dtype

    bos_token = torch.tensor(
        [[self.hp.start_speech_token]],
        dtype=torch.long,
        device=device,
    )
    bos_embed = self.speech_emb(bos_token)
    bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

    if inputs_embeds.size(0) != 2:
        raise RuntimeError(
            "Multilingual CUDA graph expects CFG batch size 2."
        )

    generated_ids = bos_token.clone()
    predicted = []

    top_p_warper = TopPLogitsWarper(top_p=top_p)
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    repetition_processor = RepetitionPenaltyLogitsProcessor(
        penalty=float(repetition_penalty)
    )

    context_len = int(inputs_embeds.shape[1])
    max_cache_len = context_len + int(max_new_tokens) + 2

    fixed_cache = _make_static_cache(
        StaticCache,
        self.cfg,
        2,
        max_cache_len,
        device,
        dtype,
    )

    # Explicit positions keep the prefill semantically aligned with the
    # dynamic-cache baseline.
    prefill_positions = torch.arange(
        context_len,
        device=device,
        dtype=torch.long,
    )

    output = self.tfmr(
        inputs_embeds=inputs_embeds,
        past_key_values=fixed_cache,
        cache_position=prefill_positions,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_states = output.last_hidden_state

    static_cache_position = torch.tensor(
        [context_len],
        device=device,
        dtype=torch.long,
    )
    static_embed = torch.zeros(
        (2, 1, self.cfg.hidden_size),
        dtype=dtype,
        device=device,
    )

    # Warm the exact one-token CUDA kernels on a separate cache. No sampling
    # occurs here, so the benchmark RNG stream remains unchanged.
    warmup_cache = _make_static_cache(
        StaticCache,
        self.cfg,
        2,
        max_cache_len,
        device,
        dtype,
    )
    warmup_position = torch.tensor(
        [context_len],
        device=device,
        dtype=torch.long,
    )
    dummy = torch.full(
        (1, 1),
        self.hp.start_speech_token,
        dtype=torch.long,
        device=device,
    )
    dummy_embed = self.speech_emb(dummy)
    dummy_embed = (
        dummy_embed
        + self.speech_pos_emb.get_fixed_embedding(1)
    )
    dummy_embed = torch.cat([dummy_embed, dummy_embed])

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_embed.copy_(dummy_embed)
            self.tfmr(
                inputs_embeds=static_embed,
                past_key_values=warmup_cache,
                cache_position=warmup_position,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            warmup_position.add_(1)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    del warmup_cache, warmup_position

    # Capture one transformer decode step. The input tensor and cache-position
    # tensor keep stable addresses and are updated in-place before each replay.
    static_embed.copy_(dummy_embed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = self.tfmr(
            inputs_embeds=static_embed,
            past_key_values=fixed_cache,
            cache_position=static_cache_position,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    static_hidden_states = captured.last_hidden_state

    for i in range(max_new_tokens):
        next_token = _sample_next(
            self,
            hidden_states,
            generated_ids,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty_processor=repetition_processor,
            min_p_warper=min_p_warper,
            top_p_warper=top_p_warper,
        )

        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if stop_on_eos and bool(
            (next_token.view(-1) == self.hp.stop_speech_token).all().item()
        ):
            break

        next_token_embed = self.speech_emb(next_token)
        next_token_embed = (
            next_token_embed
            + self.speech_pos_emb.get_fixed_embedding(i + 1)
        )
        next_token_embed = torch.cat(
            [next_token_embed, next_token_embed]
        )

        static_embed.copy_(next_token_embed)
        graph.replay()
        hidden_states = static_hidden_states
        static_cache_position.add_(1)

    torch.cuda.synchronize()

    if not predicted:
        return torch.empty(
            (1, 0),
            dtype=torch.long,
            device=device,
        )

    return torch.cat(predicted, dim=1)
