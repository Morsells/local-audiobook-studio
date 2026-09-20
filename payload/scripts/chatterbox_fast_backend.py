from __future__ import annotations

# Production Fast Quality profile selected by the full-generation benchmark.
FAST_QUALITY_PROFILE = "adaptive_v6"
ADAPTIVE_TOKENS_PER_WORD = 16
ADAPTIVE_RESERVE_TOKENS = 64
ADAPTIVE_MIN_TOKENS = 256


def _make_static_cache(
    StaticCache,
    config,
    batch_size,
    max_cache_len,
    device,
    dtype,
):
    """Support StaticCache constructor variants used by recent Transformers."""
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
        "Fast Quality CUDA-graph backend."
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


def _adaptive_generation_limit(self, requested_limit: int) -> int:
    """
    Choose a conservative speech-token limit from the current phrase length.

    If this limit is reached without EOS, inference_cuda_graph rewinds the CUDA
    RNG and repeats the phrase with the original requested limit.
    """
    words = int(getattr(self, "_las_phrase_word_count", 0) or 0)
    if words <= 0:
        return int(requested_limit)

    adaptive = max(
        ADAPTIVE_MIN_TOKENS,
        words * ADAPTIVE_TOKENS_PER_WORD + ADAPTIVE_RESERVE_TOKENS,
    )
    return min(int(requested_limit), int(adaptive))


def _decode_cuda_graph(
    self,
    *,
    t3_cond,
    text_tokens,
    initial_speech_tokens,
    limit,
    stop_on_eos,
    temperature,
    top_p,
    min_p,
    repetition_penalty,
    cfg_weight,
):
    """
    Execute one FP32 Fast Quality decode attempt.

    Sampling, CFG, model weights and downstream S3Gen semantics remain
    unchanged. This combines the production CUDA graph with the allocation
    reductions that passed the V3 parity benchmark.
    """
    import torch
    from transformers import StaticCache
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    limit = int(limit)

    embeds, _ = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
        cfg_weight=cfg_weight,
    )

    device = embeds.device
    dtype = embeds.dtype
    speech_emb = self.speech_emb
    pos_weight = self.speech_pos_emb.emb.weight
    speech_head = self.speech_head
    stop_token = self.hp.stop_speech_token

    bos_token = torch.tensor(
        [[self.hp.start_speech_token]],
        dtype=torch.long,
        device=device,
    )
    bos_embed = speech_emb(bos_token)
    bos_embed = bos_embed + pos_weight[0].view(1, 1, -1)
    inputs_embeds = torch.cat(
        [embeds, bos_embed.expand(2, -1, -1)],
        dim=1,
    )

    if inputs_embeds.size(0) != 2:
        raise RuntimeError(
            "Multilingual CUDA graph expects CFG batch size 2."
        )

    # Preallocate token history instead of copying an ever-growing torch.cat()
    # result on every generated speech token.
    generated = torch.empty(
        (1, limit + 1),
        dtype=torch.long,
        device=device,
    )
    generated[:, 0:1] = bos_token
    generated_len = 1
    predicted_len = 0
    hit_eos = False

    # Chatterbox Multilingual calls this path with top_p=1.0. At exactly 1.0
    # TopP is an identity but Transformers still sorts the full vocabulary.
    top_p_warper = (
        TopPLogitsWarper(top_p=top_p)
        if float(top_p) < 1.0
        else None
    )
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    repetition_processor = RepetitionPenaltyLogitsProcessor(
        penalty=float(repetition_penalty)
    )
    cfg_tensor = torch.as_tensor(
        cfg_weight,
        device=device,
        dtype=dtype,
    )

    context_len = int(inputs_embeds.shape[1])
    max_cache_len = context_len + limit + 2

    fixed_cache = _make_static_cache(
        StaticCache,
        self.cfg,
        2,
        max_cache_len,
        device,
        dtype,
    )

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

    # Warm the exact one-token kernels on a separate cache. No sampling occurs
    # here, so this does not consume the production RNG stream.
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
    dummy_embed = speech_emb(dummy)
    dummy_embed = dummy_embed + pos_weight[1].view(1, 1, -1)
    dummy_cfg = dummy_embed.expand(2, -1, -1)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_embed.copy_(dummy_cfg)
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

    static_embed.copy_(dummy_cfg)
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

    for i in range(limit):
        history = generated[:, :generated_len]

        logits_step = speech_head(hidden_states)[:, -1, :]
        cond = logits_step[0:1, :]
        uncond = logits_step[1:2, :]
        logits = cond + cfg_tensor * (cond - uncond)

        logits = repetition_processor(history, logits)

        if temperature != 1.0:
            logits = logits / temperature

        logits = min_p_warper(history, logits)
        if top_p_warper is not None:
            logits = top_p_warper(history, logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated[:, generated_len:generated_len + 1] = next_token
        generated_len += 1
        predicted_len += 1

        if stop_on_eos and bool(
            (next_token.view(-1) == stop_token).all().item()
        ):
            hit_eos = True
            break

        next_token_embed = speech_emb(next_token)
        next_token_embed = (
            next_token_embed
            + pos_weight[i + 1].view(1, 1, -1)
        )

        static_embed.copy_(next_token_embed.expand(2, -1, -1))
        graph.replay()
        hidden_states = static_hidden_states
        static_cache_position.add_(1)

    # No unconditional synchronize here. S3Gen consumes the result on the same
    # CUDA stream. The fallback path synchronizes before restoring the RNG.
    return (
        generated[:, 1:1 + predicted_len].clone(),
        hit_eos,
        max_cache_len,
    )


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
    Production Fast Quality / adaptive V6 decode.

    Normal audiobook phrases use a smaller StaticCache. If a phrase fails to
    emit EOS before the adaptive limit, CUDA RNG is rewound and the phrase is
    regenerated with the original limit. This preserves the baseline random
    stream instead of returning truncated speech.
    """
    import torch

    if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
        raise RuntimeError(
            "Fast Quality CUDA graph requires an NVIDIA CUDA device."
        )
    if prepend_prompt_speech_tokens is not None:
        raise AssertionError(
            "prepend_prompt_speech_tokens is not implemented"
        )
    if num_return_sequences != 1:
        raise ValueError(
            "Fast Quality supports num_return_sequences=1 only."
        )
    if not do_sample:
        raise ValueError(
            "Fast Quality currently mirrors the sampling path only."
        )

    text_tokens = _validate_inputs(self, text_tokens)
    requested_limit = int(
        max_new_tokens or self.hp.max_speech_tokens
    )

    if initial_speech_tokens is None:
        initial_speech_tokens = (
            self.hp.start_speech_token
            * torch.ones_like(text_tokens[:, :1])
        )

    adaptive_limit = _adaptive_generation_limit(
        self,
        requested_limit,
    )
    fallback_possible = adaptive_limit < requested_limit

    # Save the CUDA random stream before the adaptive attempt. If the phrase
    # reaches its cap, restore this exact state and retry at the original limit.
    rng_state = (
        torch.cuda.get_rng_state()
        if fallback_possible
        else None
    )

    tokens, hit_eos, cache_len = _decode_cuda_graph(
        self,
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        initial_speech_tokens=initial_speech_tokens,
        limit=adaptive_limit,
        stop_on_eos=stop_on_eos,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        cfg_weight=cfg_weight,
    )

    meta = {
        "profile": FAST_QUALITY_PROFILE,
        "words": int(
            getattr(self, "_las_phrase_word_count", 0) or 0
        ),
        "requested_limit": requested_limit,
        "adaptive_limit": adaptive_limit,
        "cache_len": cache_len,
        "fallback": False,
        "tokens": int(tokens.shape[-1]),
    }

    if (
        stop_on_eos
        and fallback_possible
        and not hit_eos
    ):
        # Finish outstanding GPU work before restoring the random stream.
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng_state)

        tokens, hit_eos, cache_len = _decode_cuda_graph(
            self,
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            initial_speech_tokens=initial_speech_tokens,
            limit=requested_limit,
            stop_on_eos=stop_on_eos,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            cfg_weight=cfg_weight,
        )
        meta.update({
            "cache_len": cache_len,
            "fallback": True,
            "tokens": int(tokens.shape[-1]),
        })

    self._las_fast_quality_last = meta
    return tokens
