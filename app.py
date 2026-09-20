from __future__ import annotations

import json
from pathlib import Path

from src.audiobook_studio.privacy import activate_privacy_lock

activate_privacy_lock()

import streamlit as st

from src.audiobook_studio.audio import (
    chatterbox_model_ready,
    chatterbox_server_available,
    chatterbox_server_health,
    qwen_server_available,
    qwen_server_health,
)
from src.audiobook_studio.chunker import word_count
from src.audiobook_studio.config import (
    CHATTERBOX_LANGUAGES,
    DATA_DIR,
    KOKORO_LANGUAGES,
    KOKORO_MODEL,
    KOKORO_VOICES,
    MODEL_DIR,
    QWEN_LANGUAGES,
    QWEN_SPEAKERS,
    VOICE_PRESETS,
)
from src.audiobook_studio.hardware import (
    detect_hardware,
    resolve_device,
    resolve_precision,
)
from src.audiobook_studio.local_ai import (
    list_models,
    ollama_available,
    propose_spoken_rewrite,
    review_translation,
)
from src.audiobook_studio.models import (
    BuildOptions,
    OutputMode,
    ReadingMode,
    TTSEngine,
    TranslationEngine,
)
from src.audiobook_studio.pipeline import (
    analyze_book,
    build_audio,
    estimate_stats,
    export_project_m4b,
    generate_pronunciation_test,
    generate_test_audio,
    humanize_sections,
    inspect_chunks,
    narration_sections,
    translate_sections_batch,
    translation_status,
)
from src.audiobook_studio.project import ProjectStore
from src.audiobook_studio.preview import resolve_source_preview_options
from src.audiobook_studio.queue_manager import (
    clear_finished,
    get_control,
    get_queue,
    set_control,
    set_jobs,
    start_worker,
    worker_status,
)
from src.audiobook_studio.semantic import chapter_groups
from src.audiobook_studio.source_parser import get_source_count, source_kind
from src.audiobook_studio.suspicious import find_suspicious_words
from src.audiobook_studio.storage import (
    all_projects_report,
    clear_preview_cache,
    clear_redundant_intermediate_wavs,
    clear_tts_chunk_cache,
    compact_finished_project,
    format_bytes,
    global_safe_cleanup,
    project_storage_report,
    prune_tts_chunk_cache,
    safe_cleanup_project,
)


st.set_page_config(
    page_title="Local Audiobook Studio",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        padding: 0.75rem 0.85rem;
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Local Audiobook Studio")
st.caption(
    "Local book → cleanup → optional translation → natural TTS → audiobook"
)


def as_records(value):
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return list(value)


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "Calibrates after generated chunks"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def parse_timestamp(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError("Use seconds, MM:SS, or HH:MM:SS")



def make_live_progress_ui():
    bar = st.progress(0.0)
    label = st.empty()

    def callback(message: str, fraction: float) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        label.write(message)
        bar.progress(fraction)

    return callback, bar, label


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def render_live_queue_status(project_dir: str) -> None:
        store = ProjectStore.from_existing_dir(project_dir)
        worker = worker_status(store)
        queue = get_queue(store)
        jobs = queue.get("jobs", [])
        done = sum(1 for job in jobs if job.get("status") == "done")
        failed = sum(1 for job in jobs if job.get("status") == "failed")
        total = len(jobs)

        remaining_seconds = 0.0
        has_estimates = False
        current_job = str(worker.get("current_job", "") or "")
        current_fraction = max(
            0.0,
            min(1.0, float(worker.get("progress", 0.0) or 0.0)),
        )

        for job in jobs:
            estimate = float(job.get("estimated_seconds", 0) or 0)
            if estimate <= 0:
                continue
            has_estimates = True
            status_name = str(job.get("status", "waiting"))
            if status_name in {"waiting", "retry"}:
                remaining_seconds += estimate
            elif status_name == "running":
                if str(job.get("id", "")) == current_job:
                    remaining_seconds += estimate * (1.0 - current_fraction)
                else:
                    remaining_seconds += estimate

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queue", f"{done}/{total} complete" if total else "empty")
        c2.metric("Failed", failed)
        c3.metric("Worker", "running" if worker.get("active") else "stopped")
        c4.metric(
            "Queue ETA",
            f"~{fmt_seconds(remaining_seconds)}"
            if has_estimates and remaining_seconds > 0
            else ("done" if total and done == total else "calibrating"),
        )

        message = str(worker.get("message", "") or "")
        if message:
            st.caption(message)

        if worker.get("active"):
            fraction = max(0.0, min(1.0, float(worker.get("progress", 0.0) or 0.0)))
            st.progress(fraction)

else:
    def render_live_queue_status(project_dir: str) -> None:
        store = ProjectStore.from_existing_dir(project_dir)
        worker = worker_status(store)
        queue = get_queue(store)
        jobs = queue.get("jobs", [])
        done = sum(1 for job in jobs if job.get("status") == "done")
        st.write(
            f"Queue: **{done}/{len(jobs)} complete** · "
            f"Worker: **{'running' if worker.get('active') else 'stopped'}**"
        )
        if worker.get("message"):
            st.caption(str(worker.get("message")))
        if worker.get("active"):
            st.progress(max(0.0, min(1.0, float(worker.get("progress", 0.0) or 0.0))))


def tts_is_ready(options: BuildOptions) -> bool:
    if options.tts_engine == TTSEngine.QWEN3:
        return qwen_server_available(options.qwen_server_url)
    if options.tts_engine == TTSEngine.CHATTERBOX:
        return (
            chatterbox_model_ready()
            and chatterbox_server_available(options.chatterbox_server_url)
        )
    return KOKORO_MODEL.exists() and KOKORO_VOICES.exists()


def selected_sections_from_state(sections):
    selected_ids = set(st.session_state.get("selected_section_ids", []))
    return [section for section in sections if section.id in selected_ids]


hardware_info = detect_hardware()


with st.sidebar:
    st.subheader("Status")

    if chatterbox_server_available():
        st.success("Chatterbox V3 · ready")
        cb_sidebar_health = chatterbox_server_health()
        if cb_sidebar_health:
            backend_name = (
                "Fast Quality"
                if cb_sidebar_health.get("fast_quality_cuda_graph", False)
                else "Official"
            )
            st.caption(
                f"{backend_name} · {cb_sidebar_health.get('device', 'local')}"
            )
    elif chatterbox_model_ready():
        st.warning("Chatterbox V3 · model installed")
        st.caption("Sidecar is not running.")
    else:
        st.error("Chatterbox V3 · missing")

    if KOKORO_MODEL.exists() and KOKORO_VOICES.exists():
        st.success("Kokoro · ready")
        st.caption("Lightweight English narration")
    else:
        st.warning("Kokoro · model missing")

    with st.expander("Optional engine"):
        if qwen_server_available():
            st.success("Qwen3-TTS · running")
        else:
            st.caption("Qwen3-TTS · not running")
        st.caption(
            "Kept as an optional fallback. It is not needed for the normal "
            "Chatterbox workflow."
        )

    with st.expander("System"):
        if hardware_info.gpu_name:
            st.write(f"**GPU:** {hardware_info.gpu_name}")
            if hardware_info.gpu_vram_gb:
                st.caption(f"{hardware_info.gpu_vram_gb:.1f} GB VRAM")
        else:
            st.write("**GPU:** not detected")

        if hardware_info.cuda_available:
            st.success(
                f"CUDA ready · {hardware_info.recommended_precision}"
            )
        else:
            st.caption("Main Python environment is using CPU.")

        if ollama_available():
            st.success("Ollama · local")
        else:
            st.caption("Ollama · not running")

    st.divider()
    st.caption("🔒 Local-only runtime · loopback sockets · telemetry off")


st.subheader("Open a book")
st.caption("PDF · EPUB · TXT · Markdown · processed locally")

uploaded = st.file_uploader(
    "Book file",
    type=["pdf", "epub", "txt", "md"],
    label_visibility="collapsed",
)

if not uploaded:
    st.stop()

project = ProjectStore(uploaded.getvalue(), uploaded.name)
project_key = project.pdf_hash
kind = source_kind(project.source_path)

try:
    source_count = get_source_count(project.source_path)
except Exception as exc:
    st.exception(exc)
    st.stop()

if st.session_state.get("active_project_key") != project_key:
    for key in list(st.session_state.keys()):
        if key.startswith("book_") or key in {
            "sections",
            "current_options",
            "selected_section_ids",
            "voice_auditions",
            "ai_proposal",
            "translation_review",
        }:
            del st.session_state[key]
    st.session_state.active_project_key = project_key

saved_settings = project.load_settings()
_, _, saved_sections = project.load_analysis()

if "sections" not in st.session_state:
    st.session_state.sections = saved_sections

if "selected_section_ids" not in st.session_state:
    st.session_state.selected_section_ids = saved_settings.get(
        "selected_section_ids",
        [section.id for section in saved_sections],
    )

unit_word = "PDF pages" if kind == "PDF" else "logical units"
st.success(
    f"Project: **{uploaded.name}** · {kind} · {source_count} {unit_word} · "
    "project data stays under `.audiobook_data/`."
)

(
    tab_setup,
    tab_blocks,
    tab_translation,
    tab_language,
    tab_queue,
    tab_chunks,
    tab_export,
    tab_ai,
    tab_project,
) = st.tabs([
    "1. Setup",
    "2. Blocks & Edit",
    "3. Translate",
    "4. Pronunciation & Rules",
    "5. Queue & Generate",
    "6. Chunk Inspector",
    "7. Listen & Export",
    "8. Local AI Review",
    "9. Project & Privacy",
])


with tab_setup:
    st.subheader("Source range")

    default_start = int(saved_settings.get("start_page", 1))
    default_end = min(
        source_count,
        max(1, int(saved_settings.get("end_page", source_count))),
    )

    left, right = st.columns(2)

    with left:
        start_page = st.number_input(
            "Start PDF page" if kind == "PDF" else "Start logical unit",
            1,
            source_count,
            min(source_count, max(1, default_start)),
            1,
        )
        end_page = st.number_input(
            "End PDF page" if kind == "PDF" else "End logical unit",
            1,
            source_count,
            default_end,
            1,
        )

        saved_mode = saved_settings.get(
            "reading_mode",
            ReadingMode.NATURAL.value,
        )
        reading_mode = st.radio(
            "Reading mode",
            [mode.value for mode in ReadingMode],
            index=(
                [mode.value for mode in ReadingMode].index(saved_mode)
                if saved_mode in [mode.value for mode in ReadingMode]
                else 1
            ),
            horizontal=True,
        )

        saved_output = saved_settings.get(
            "output_mode",
            OutputMode.CHAPTER.value,
        )
        output_mode = st.selectbox(
            "Final audio structure",
            [mode.value for mode in OutputMode],
            index=(
                [mode.value for mode in OutputMode].index(saved_output)
                if saved_output in [mode.value for mode in OutputMode]
                else 1
            ),
        )

        ocr_enabled = False
        ocr_language = "eng"
        if kind == "PDF":
            ocr_enabled = st.checkbox(
                "OCR fallback for scanned pages",
                value=bool(saved_settings.get("ocr_enabled", False)),
            )
            ocr_language = st.text_input(
                "OCR language code",
                value=str(saved_settings.get("ocr_language", "eng")),
                disabled=not ocr_enabled,
                help="Examples: eng, deu, eng+deu",
            )

    with right:
        visible_tts_engines = [
            TTSEngine.CHATTERBOX,
            TTSEngine.KOKORO,
            TTSEngine.QWEN3,
        ]
        tts_labels = {
            TTSEngine.CHATTERBOX: "Chatterbox V3 — German / multilingual (recommended)",
            TTSEngine.KOKORO: "Kokoro — lightweight English",
            TTSEngine.QWEN3: "Qwen3-TTS — optional fallback",
        }

        saved_tts = saved_settings.get(
            "tts_engine",
            TTSEngine.CHATTERBOX.value,
        )
        saved_visible_engine = next(
            (
                engine
                for engine in visible_tts_engines
                if engine.value == saved_tts
            ),
            TTSEngine.CHATTERBOX,
        )

        tts_engine = st.selectbox(
            "Narration engine",
            visible_tts_engines,
            index=visible_tts_engines.index(saved_visible_engine),
            format_func=lambda engine: tts_labels[engine],
        )

        voice = saved_settings.get("voice", "af_heart")
        language = saved_settings.get("language", "en-us")
        qwen_server_url = str(
            saved_settings.get("qwen_server_url", "http://127.0.0.1:7867")
        )
        qwen_mode = str(saved_settings.get("qwen_mode", "custom_voice"))
        qwen_speaker = str(saved_settings.get("qwen_speaker", "Ryan"))
        qwen_instruct = str(
            saved_settings.get(
                "qwen_instruct",
                "Calm, natural, authoritative audiobook narration. "
                "Clear pronunciation and moderate pacing.",
            )
        )

        chatterbox_server_url = str(
            saved_settings.get("chatterbox_server_url", "http://127.0.0.1:7869")
        )
        chatterbox_reference_path = str(
            saved_settings.get("chatterbox_reference_path", "")
        )
        chatterbox_inference_backend = str(
            saved_settings.get("chatterbox_inference_backend", "auto")
        ).strip().lower()
        chatterbox_exaggeration = float(
            saved_settings.get("chatterbox_exaggeration", 0.5)
        )
        chatterbox_cfg_weight = float(
            saved_settings.get("chatterbox_cfg_weight", 0.5)
        )
        chatterbox_temperature = float(
            saved_settings.get("chatterbox_temperature", 0.65)
        )
        chatterbox_repetition_penalty = float(
            saved_settings.get("chatterbox_repetition_penalty", 1.15)
        )
        chatterbox_audiobook_pacing = bool(
            saved_settings.get("chatterbox_audiobook_pacing", True)
        )
        chatterbox_max_phrase_words = int(
            saved_settings.get("chatterbox_max_phrase_words", 36)
        )
        chatterbox_comma_pause_ms = int(
            saved_settings.get("chatterbox_comma_pause_ms", 170)
        )
        chatterbox_semicolon_pause_ms = int(
            saved_settings.get("chatterbox_semicolon_pause_ms", 320)
        )
        chatterbox_sentence_pause_ms = int(
            saved_settings.get("chatterbox_sentence_pause_ms", 520)
        )

        if tts_engine == TTSEngine.KOKORO:
            lang_names = list(KOKORO_LANGUAGES.keys())
            current_lang = next(
                (
                    name
                    for name, code in KOKORO_LANGUAGES.items()
                    if code == language
                ),
                lang_names[0],
            )
            lang_name = st.selectbox(
                "Narration language",
                lang_names,
                index=lang_names.index(current_lang),
            )
            language = KOKORO_LANGUAGES[lang_name]

            voice_values = list(VOICE_PRESETS.values())
            voice_names = list(VOICE_PRESETS.keys())
            voice_index = voice_values.index(voice) if voice in voice_values else 0
            voice_name = st.selectbox("Voice", voice_names, index=voice_index)
            voice = VOICE_PRESETS[voice_name]

            if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
                st.warning("Run `python scripts/download_models.py` once.")

        elif tts_engine == TTSEngine.CHATTERBOX:
            voice = "chatterbox-multilingual-v3"

            chatterbox_server_url = st.text_input(
                "Local Chatterbox server URL",
                value=chatterbox_server_url,
                help="Localhost only. Book text stays on this machine.",
            )

            cb_lang_names = list(CHATTERBOX_LANGUAGES.keys())
            current_cb_lang = next(
                (
                    name
                    for name, code in CHATTERBOX_LANGUAGES.items()
                    if code == language
                ),
                "German",
            )
            cb_lang_name = st.selectbox(
                "Narration language",
                cb_lang_names,
                index=cb_lang_names.index(current_cb_lang),
            )
            language = CHATTERBOX_LANGUAGES[cb_lang_name]

            cb_health_now = (
                chatterbox_server_health(chatterbox_server_url)
                if chatterbox_server_available(chatterbox_server_url)
                else {}
            )
            backend_labels = {
                "auto": "Auto — Fast Quality when supported (recommended)",
                "cuda_graph": "Fast Quality — CUDA Graph",
                "official": "Official Quality — compatibility",
            }
            if chatterbox_inference_backend not in backend_labels:
                chatterbox_inference_backend = "auto"

            chatterbox_inference_backend = st.selectbox(
                "Inference / speed mode",
                list(backend_labels.keys()),
                index=list(backend_labels.keys()).index(
                    chatterbox_inference_backend
                ),
                format_func=lambda value: backend_labels[value],
                help=(
                    "Fast Quality uses the same Chatterbox Multilingual V3 model, "
                    "voice reference and sampling settings, but accelerates the T3 "
                    "decode with CUDA Graphs. Official keeps the original inference path."
                ),
            )

            cuda_graph_ready = bool(
                cb_health_now.get("fast_quality_cuda_graph", False)
            )
            resolved_ui_backend = (
                "cuda_graph"
                if chatterbox_inference_backend == "auto" and cuda_graph_ready
                else (
                    "official"
                    if chatterbox_inference_backend == "auto"
                    else chatterbox_inference_backend
                )
            )

            if resolved_ui_backend == "cuda_graph":
                st.success(
                    "Fast Quality active: CUDA Graph. Same V3 voice/settings; "
                    "validated production CUDA Graph path."
                )
            elif chatterbox_inference_backend == "auto":
                st.info(
                    "Auto resolved to Official because CUDA Graph is not available "
                    "from the running sidecar."
                )
            else:
                st.caption("Official Chatterbox inference / compatibility mode.")

            if (
                chatterbox_inference_backend == "cuda_graph"
                and not cuda_graph_ready
            ):
                st.warning(
                    "CUDA Graph was selected but the running sidecar does not expose it. "
                    "Restart Chatterbox or choose Auto / Official."
                )

            st.markdown("**Voice / accent reference**")
            st.caption(
                "For clean Hochdeutsch, use a clean German reference clip from a voice "
                "you have permission to use. The clip is copied into this local project "
                "and never uploaded during normal runtime."
            )

            existing_reference = (
                Path(chatterbox_reference_path)
                if chatterbox_reference_path
                else None
            )
            if existing_reference and existing_reference.exists():
                st.success(f"Reference voice: `{existing_reference.name}`")
            elif chatterbox_reference_path:
                st.warning(
                    "The saved Chatterbox reference file no longer exists. "
                    "Upload it again or use the built-in voice."
                )
                chatterbox_reference_path = ""

            uploaded_reference = st.file_uploader(
                "Optional German reference voice",
                type=["wav", "mp3", "flac", "m4a", "ogg"],
                key=f"chatterbox_reference_{project_key}",
                help=(
                    "A same-language reference reduces unwanted transferred accents. "
                    "Leave empty to use Chatterbox's built-in conditioning voice."
                ),
            )

            if uploaded_reference is not None:
                ref_dir = project.project_dir / "voice_refs"
                ref_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(uploaded_reference.name).suffix.lower() or ".wav"
                ref_path = ref_dir / f"chatterbox_reference{suffix}"
                ref_path.write_bytes(uploaded_reference.getvalue())
                chatterbox_reference_path = str(ref_path.resolve())
                st.success(
                    f"Stored locally: `{ref_path.relative_to(project.project_dir)}`"
                )

            if chatterbox_reference_path and st.button(
                "Remove Chatterbox reference voice",
                use_container_width=True,
            ):
                try:
                    ref_path = Path(chatterbox_reference_path)
                    if (
                        ref_path.exists()
                        and project.project_dir.resolve() in ref_path.resolve().parents
                    ):
                        ref_path.unlink(missing_ok=True)
                except Exception:
                    pass
                chatterbox_reference_path = ""
                st.success("Reference removed. Built-in voice will be used.")

            with st.expander("Chatterbox narration tuning", expanded=True):
                chatterbox_audiobook_pacing = st.checkbox(
                    "Audiobook pacing mode",
                    value=bool(chatterbox_audiobook_pacing),
                    help=(
                        "Recommended. Synthesizes shorter punctuation-aware phrases and "
                        "inserts real audio pauses after commas, semicolons and sentences. "
                        "This also reduces skipped words on long prompts."
                    ),
                )

                if chatterbox_audiobook_pacing:
                    st.caption(
                        "Audiobook mode prioritizes faithful reading and breathing room "
                        "over maximum generation speed."
                    )
                    chatterbox_max_phrase_words = st.slider(
                        "Maximum words per synthesis phrase",
                        18,
                        60,
                        int(chatterbox_max_phrase_words),
                        2,
                        help=(
                            "Shorter phrases reduce omissions. 30–40 works well for books."
                        ),
                    )

                    p1, p2, p3 = st.columns(3)
                    with p1:
                        chatterbox_comma_pause_ms = st.number_input(
                            "Comma pause (ms)",
                            min_value=0,
                            max_value=800,
                            value=int(chatterbox_comma_pause_ms),
                            step=10,
                        )
                    with p2:
                        chatterbox_semicolon_pause_ms = st.number_input(
                            "Semicolon / colon pause (ms)",
                            min_value=0,
                            max_value=1200,
                            value=int(chatterbox_semicolon_pause_ms),
                            step=10,
                        )
                    with p3:
                        chatterbox_sentence_pause_ms = st.number_input(
                            "Sentence pause (ms)",
                            min_value=0,
                            max_value=1600,
                            value=int(chatterbox_sentence_pause_ms),
                            step=10,
                        )

                chatterbox_exaggeration = st.slider(
                    "Expressiveness",
                    0.0,
                    1.0,
                    float(chatterbox_exaggeration),
                    0.05,
                    help="0.5 is a good neutral narration starting point.",
                )
                chatterbox_cfg_weight = st.slider(
                    "Voice/reference adherence (CFG)",
                    0.0,
                    1.0,
                    float(chatterbox_cfg_weight),
                    0.05,
                    help=(
                        "0.5 is a good default. If a reference clip speaks too quickly, "
                        "around 0.3 can improve pacing."
                    ),
                )
                chatterbox_temperature = st.slider(
                    "Generation temperature",
                    0.3,
                    1.2,
                    float(chatterbox_temperature),
                    0.05,
                    help=(
                        "For books, 0.60–0.70 is recommended. Lower values generally "
                        "favor faithful wording over creative prosody."
                    ),
                )
                chatterbox_repetition_penalty = st.slider(
                    "Repetition penalty",
                    1.0,
                    1.5,
                    float(chatterbox_repetition_penalty),
                    0.05,
                    help="For audiobook narration, around 1.10–1.20 is a good range.",
                )

            if not chatterbox_model_ready():
                st.warning("Chatterbox Multilingual V3 is not installed yet.")
                st.code(
                    ".\\\\Setup-Chatterbox-GPU.bat",
                    language="powershell",
                )
            elif chatterbox_server_available(chatterbox_server_url):
                cb_health = chatterbox_server_health(chatterbox_server_url)
                st.success(
                    "Chatterbox Multilingual V3 local sidecar ready."
                )
                if cb_health:
                    st.caption(
                        f"Device: {cb_health.get('device', '-')} · "
                        f"GPU: {cb_health.get('gpu_name', '-')} · "
                        f"runtime: {cb_health.get('runtime', 'local')}"
                    )
            else:
                st.info(
                    "Chatterbox is installed but its sidecar is not running. "
                    "Use Start-Audiobook-Studio-With-Chatterbox.bat."
                )

            st.caption(
                "Generated Chatterbox audio contains Resemble AI's imperceptible "
                "PerTh watermark. The watermark is embedded locally; it is not an upload."
            )

        else:
            qwen_server_url = st.text_input(
                "Local Qwen server URL",
                value=qwen_server_url,
                help="Privacy Lock accepts localhost/127.0.0.1 only.",
            )

            lang_names = list(QWEN_LANGUAGES.keys())
            current_lang = next(
                (
                    name
                    for name, code in QWEN_LANGUAGES.items()
                    if code == language
                ),
                "German" if str(language).startswith("de") else "English",
            )
            qwen_lang_name = st.selectbox(
                "Narration language",
                lang_names,
                index=lang_names.index(current_lang),
            )
            language = QWEN_LANGUAGES[qwen_lang_name]

            qwen_mode = st.selectbox(
                "Qwen mode",
                ["custom_voice", "voice_design"],
                index=0 if qwen_mode != "voice_design" else 1,
            )

            if qwen_mode == "custom_voice":
                qwen_speaker = st.selectbox(
                    "Qwen speaker",
                    QWEN_SPEAKERS,
                    index=(
                        QWEN_SPEAKERS.index(qwen_speaker)
                        if qwen_speaker in QWEN_SPEAKERS
                        else 0
                    ),
                )

            qwen_instruct = st.text_area(
                "Voice / narration instruction",
                value=qwen_instruct,
                height=100,
            )

            if qwen_server_available(qwen_server_url):
                qwen_health = qwen_server_health(qwen_server_url)
                st.success("Local Qwen3-TTS server detected.")
                if qwen_health:
                    st.caption(
                        f"Device: {qwen_health.get('device', '-')} · "
                        f"precision: {qwen_health.get('precision', '-')} · "
                        f"GPU: {qwen_health.get('gpu_name', '-')}"
                    )
            else:
                st.warning("Qwen server is not running yet.")
                st.code(
                    ".\\Start-Audiobook-Studio-With-Qwen.bat",
                    language="powershell",
                )
                st.caption(
                    "The Qwen sidecar automatically uses FP16 on RTX 20xx/Turing "
                    "and native BF16 only on hardware that supports it."
                )

        speed = st.slider(
            "Narration speed",
            0.75,
            1.50,
            float(saved_settings.get("speed", 1.0)),
            0.05,
        )

    st.subheader("Performance & hardware")
    perf1, perf2 = st.columns(2)

    with perf1:
        hardware_labels = {
            "auto": "Auto — use the fastest safe device",
            "gpu": "GPU preferred — require CUDA",
            "cpu": "CPU only",
        }
        saved_hardware_mode = str(saved_settings.get("hardware_mode", "auto"))
        if saved_hardware_mode not in hardware_labels:
            saved_hardware_mode = "auto"

        hardware_mode = st.selectbox(
            "Main app / OPUS compute mode",
            list(hardware_labels.keys()),
            index=list(hardware_labels.keys()).index(saved_hardware_mode),
            format_func=lambda value: hardware_labels[value],
        )

        translation_batch_size = st.number_input(
            "OPUS translation batch size",
            min_value=0,
            max_value=32,
            value=int(saved_settings.get("translation_batch_size", 0)),
            step=1,
            help=(
                "0 = automatic. The app chooses a batch size from available VRAM/CPU. "
                "For an 11 GB GPU the normal automatic target is 8."
            ),
        )

    with perf2:
        st.markdown("**Detected hardware**")
        st.write(f"CPU: {hardware_info.cpu_name} · {hardware_info.logical_cpus} logical threads")

        if hardware_info.gpu_name:
            st.write(
                f"GPU: {hardware_info.gpu_name}"
                + (
                    f" · {hardware_info.gpu_vram_gb:.1f} GB VRAM"
                    if hardware_info.gpu_vram_gb
                    else ""
                )
            )
        else:
            st.write("GPU: not detected")

        if hardware_info.cuda_available:
            try:
                selected_device = resolve_device(hardware_mode, hardware_info)
                selected_precision = resolve_precision(hardware_mode, hardware_info)
                selected_batch = int(
                    translation_batch_size
                    or hardware_info.recommended_translation_batch
                )
                st.success(
                    f"Auto accelerator: {selected_device.upper()} · "
                    f"{selected_precision} · OPUS batch {selected_batch}"
                )
            except Exception as exc:
                st.error(str(exc))
        elif hardware_info.gpu_name:
            st.warning(
                "Your NVIDIA GPU is visible, but CUDA Torch is not active in the main .venv. "
                "Run `Setup-GPU-Acceleration.bat` once."
            )
        else:
            st.info("No CUDA GPU detected in this environment; CPU optimization will be used.")

    st.subheader("Internal TTS chunks")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_chunk = st.number_input(
            "Minimum words",
            40,
            160,
            int(saved_settings.get("min_chunk_words", 80)),
            5,
        )
    with c2:
        target_chunk = st.number_input(
            "Target words",
            80,
            220,
            int(saved_settings.get("target_chunk_words", 155)),
            5,
        )
    with c3:
        max_chunk = st.number_input(
            "Maximum words",
            120,
            260,
            int(saved_settings.get("max_chunk_words", 195)),
            5,
        )

    st.subheader("Natural pauses")
    p1, p2, p3 = st.columns(3)
    with p1:
        chunk_pause = st.number_input(
            "Between TTS chunks (ms)",
            0,
            1500,
            int(saved_settings.get("chunk_pause_ms", 250)),
            50,
        )
    with p2:
        section_pause = st.number_input(
            "Between sections (ms)",
            0,
            3000,
            int(saved_settings.get("section_pause_ms", 700)),
            50,
        )
    with p3:
        chapter_pause = st.number_input(
            "Between chapters (ms)",
            0,
            5000,
            int(saved_settings.get("chapter_pause_ms", 1400)),
            100,
        )

    translation_settings = project.load_translation_settings()
    saved_translation_engine = saved_settings.get(
        "translation_engine",
        translation_settings.get("engine", TranslationEngine.OPUS.value),
    )

    options = BuildOptions(
        start_page=int(start_page),
        end_page=int(end_page),
        voice=str(voice),
        language=str(language),
        speed=float(speed),
        reading_mode=ReadingMode(reading_mode),
        output_mode=OutputMode(output_mode),
        ocr_enabled=bool(ocr_enabled),
        ocr_language=ocr_language.strip() or "eng",
        target_chunk_words=int(target_chunk),
        min_chunk_words=int(min_chunk),
        max_chunk_words=int(max_chunk),
        chunk_pause_ms=int(chunk_pause),
        section_pause_ms=int(section_pause),
        chapter_pause_ms=int(chapter_pause),
        pronunciations=project.load_pronunciations(),
        replacement_rules=project.load_replacement_rules(),
        skip_strings=project.load_skip_strings(),
        translation_enabled=bool(
            saved_settings.get(
                "translation_enabled",
                translation_settings.get("enabled", False),
            )
        ),
        translation_engine=TranslationEngine(saved_translation_engine),
        translation_source_language=str(
            saved_settings.get(
                "translation_source_language",
                translation_settings.get("source_language", "en"),
            )
        ),
        translation_target_language=str(
            saved_settings.get(
                "translation_target_language",
                translation_settings.get("target_language", "de"),
            )
        ),
        translation_model=str(
            saved_settings.get(
                "translation_model",
                translation_settings.get("model", ""),
            )
        ),
        translation_ollama_model=str(
            saved_settings.get(
                "translation_ollama_model",
                translation_settings.get("ollama_model", ""),
            )
        ),
        translation_batch_size=int(translation_batch_size),
        hardware_mode=str(hardware_mode),
        tts_engine=tts_engine,
        qwen_server_url=qwen_server_url,
        qwen_mode=qwen_mode,
        qwen_speaker=qwen_speaker,
        qwen_instruct=qwen_instruct,
        qwen_model_label="Qwen3-TTS local sidecar",
        chatterbox_server_url=chatterbox_server_url,
        chatterbox_reference_path=chatterbox_reference_path,
        chatterbox_inference_backend=str(
            chatterbox_inference_backend
        ),
        chatterbox_exaggeration=float(chatterbox_exaggeration),
        chatterbox_cfg_weight=float(chatterbox_cfg_weight),
        chatterbox_temperature=float(chatterbox_temperature),
        chatterbox_repetition_penalty=float(
            chatterbox_repetition_penalty
        ),
        chatterbox_audiobook_pacing=bool(
            chatterbox_audiobook_pacing
        ),
        chatterbox_max_phrase_words=int(
            chatterbox_max_phrase_words
        ),
        chatterbox_comma_pause_ms=int(
            chatterbox_comma_pause_ms
        ),
        chatterbox_semicolon_pause_ms=int(
            chatterbox_semicolon_pause_ms
        ),
        chatterbox_sentence_pause_ms=int(
            chatterbox_sentence_pause_ms
        ),
    )
    st.session_state.current_options = options

    valid = start_page <= end_page and min_chunk <= target_chunk <= max_chunk
    if not valid:
        st.error("Required: start ≤ end and minimum chunk ≤ target ≤ maximum.")

    a1, a2 = st.columns(2)
    with a1:
        if st.button(
            "Analyze selected content",
            type="primary",
            use_container_width=True,
            disabled=not valid,
        ):
            with st.spinner("Extracting structure and detecting semantic blocks..."):
                try:
                    pages, sections = analyze_book(project.source_path, options)
                    sig = json.dumps(
                        {
                            "source": project.source_path.name,
                            "start": options.start_page,
                            "end": options.end_page,
                            "ocr": options.ocr_enabled,
                            "ocr_lang": options.ocr_language,
                        },
                        sort_keys=True,
                    )
                    project.save_analysis(pages, sections, sig)
                    st.session_state.sections = sections
                    st.session_state.selected_section_ids = [s.id for s in sections]
                    settings = options.to_dict()
                    settings["selected_section_ids"] = st.session_state.selected_section_ids
                    project.save_settings(settings)
                    st.success(
                        f"Detected {len(sections)} section(s) across "
                        f"{len(chapter_groups(sections))} chapter group(s)."
                    )
                except Exception as exc:
                    st.exception(exc)

    with a2:
        if st.button("Save current setup", use_container_width=True):
            settings = options.to_dict()
            settings["selected_section_ids"] = st.session_state.selected_section_ids
            project.save_settings(settings)
            st.success("Setup saved.")


with tab_blocks:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    if not sections or options is None:
        st.info("Analyze the selected content first.")
    else:
        options.pronunciations = project.load_pronunciations()
        options.replacement_rules = project.load_replacement_rules()
        options.skip_strings = project.load_skip_strings()

        st.subheader("Chapter hierarchy")
        st.dataframe(
            [
                {
                    "Chapter": title,
                    "Sections": len(members),
                    "Range": f"{min(s.start_page for s in members)}-{max(s.end_page for s in members)}",
                }
                for _, title, members in chapter_groups(sections)
            ],
            use_container_width=True,
            hide_index=True,
        )

        labels = []
        label_to_section = {}
        for section in sections:
            indent = "· " * max(0, section.level - 1)
            label = f"{indent}{section.title} — {section.start_page}-{section.end_page}"
            labels.append(label)
            label_to_section[label] = section

        selected_set = set(st.session_state.selected_section_ids)
        default_labels = [
            label for label in labels if label_to_section[label].id in selected_set
        ]
        chosen = st.multiselect("Include these sections", labels, default=default_labels)
        selected_sections = [label_to_section[label] for label in chosen]
        st.session_state.selected_section_ids = [section.id for section in selected_sections]

        settings = project.load_settings()
        settings["selected_section_ids"] = st.session_state.selected_section_ids
        project.save_settings(settings)

        if selected_sections:
            preview_section = st.selectbox(
                "Section to inspect",
                selected_sections,
                format_func=lambda section: f"{section.chapter_title or section.title} → {section.title}",
            )
            chapter_preview = st.checkbox("Preview whole chapter", value=False)

            if chapter_preview:
                preview_sections = [
                    section
                    for section in selected_sections
                    if (section.chapter_id or section.id)
                    == (preview_section.chapter_id or preview_section.id)
                ]
            else:
                preview_sections = [preview_section]

            prepared = humanize_sections(
                preview_sections,
                options,
                project.load_overrides(),
            )
            cleaned_text = "\n\n".join(text for _, text in prepared)
            raw_text = "\n\n".join(section.raw_text for section in preview_sections)

            st.subheader("Raw source vs cleaned narration")
            raw_col, clean_col = st.columns(2)

            with raw_col:
                st.text_area(
                    "Raw extracted text",
                    raw_text[:24000],
                    height=430,
                    disabled=True,
                )

            with clean_col:
                if chapter_preview:
                    st.text_area(
                        "Cleaned narration",
                        cleaned_text[:24000],
                        height=430,
                        disabled=True,
                    )
                else:
                    overrides = project.load_overrides()
                    edited_text = st.text_area(
                        "Editable cleaned narration",
                        value=(overrides.get(preview_section.id) or cleaned_text)[:30000],
                        height=430,
                    )

            if not chapter_preview:
                source_language = str(
                    project.load_translation_settings().get(
                        "source_language",
                        options.translation_source_language or "en",
                    )
                )
                source_preview_opts, source_preview_label = resolve_source_preview_options(
                    options,
                    source_language,
                    chatterbox_available=chatterbox_server_available(
                        options.chatterbox_server_url
                    ),
                )

                st.caption(
                    "Source preview engine: "
                    f"**{source_preview_label}**. "
                    "The final audiobook engine is used only for the final/translated narration."
                )

                e1, e2, e3 = st.columns(3)

                with e1:
                    if st.button("Save narration override", use_container_width=True):
                        project.save_override(preview_section.id, edited_text)
                        st.success("Override saved.")

                with e2:
                    if st.button("Reset override", use_container_width=True):
                        project.save_override(preview_section.id, "")
                        st.success("Override removed.")

                with e3:
                    source_preview_ready = tts_is_ready(source_preview_opts)
                    if st.button(
                        "Generate source-language test audio",
                        use_container_width=True,
                        disabled=not source_preview_ready,
                    ):
                        try:
                            callback, preview_bar, preview_label = make_live_progress_ui()
                            path = generate_test_audio(
                                project.project_dir,
                                preview_section,
                                edited_text or cleaned_text,
                                source_preview_opts,
                                progress=callback,
                            )
                            preview_bar.progress(1.0)
                            preview_label.success(
                                f"Source preview ready with {source_preview_label}."
                            )
                            st.session_state.book_preview_audio = str(path)
                        except Exception as exc:
                            st.exception(exc)

                if not tts_is_ready(source_preview_opts):
                    st.warning(
                        f"{source_preview_label} is not ready, so source-text preview is disabled. "
                        "This does not affect the final audiobook engine."
                    )

                preview_audio = st.session_state.get("book_preview_audio")
                if preview_audio and Path(preview_audio).exists():
                    st.audio(Path(preview_audio).read_bytes(), format="audio/wav")


with tab_translation:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    if not sections or options is None:
        st.info("Analyze the book first.")
    else:
        st.subheader("Local translation")
        st.caption(
            "No DeepL, Google Translate, or cloud API. "
            "OPUS-MT, Argos, and Ollama are local after setup."
        )

        translation_enabled = st.checkbox(
            "Use translated text for the audiobook",
            value=bool(options.translation_enabled),
        )

        t1, t2, t3 = st.columns(3)
        with t1:
            source_language = st.selectbox(
                "Source language",
                ["en", "de"],
                index=0 if options.translation_source_language == "en" else 1,
                format_func=lambda value: {"en": "English", "de": "German"}[value],
            )
        with t2:
            target_language = st.selectbox(
                "Target language",
                ["de", "en"],
                index=0 if options.translation_target_language == "de" else 1,
                format_func=lambda value: {"en": "English", "de": "German"}[value],
            )
        with t3:
            engine_value = st.selectbox(
                "Translation engine",
                [engine.value for engine in TranslationEngine],
                index=[engine.value for engine in TranslationEngine].index(
                    options.translation_engine.value
                ),
            )
            translation_engine = TranslationEngine(engine_value)

        if source_language == target_language:
            st.warning("Source and target language are the same.")

        options.translation_enabled = translation_enabled
        options.translation_source_language = source_language
        options.translation_target_language = target_language
        options.translation_engine = translation_engine

        if translation_enabled and options.tts_engine == TTSEngine.KOKORO and target_language != "en":
            st.error(
                "Kokoro is the lightweight English backend. "
                "For German translated narration, choose Chatterbox V3 "
                "(recommended) or Qwen3-TTS as an optional fallback."
            )

        if (
            options.tts_engine in {TTSEngine.QWEN3, TTSEngine.CHATTERBOX}
            and translation_enabled
        ):
            options.language = target_language

        if translation_engine == TranslationEngine.OPUS:
            st.caption(
                "For large whole-book OPUS jobs, use "
                "`Start-Audiobook-Studio-Translate.bat` so TTS sidecars "
                "stay unloaded and the GPU is dedicated to translation."
            )
            model_dir = MODEL_DIR / "translation" / f"opus-mt-{source_language}-{target_language}"
            options.translation_model = model_dir.name
            if model_dir.exists():
                st.success(f"Local OPUS model ready: `{model_dir.name}`")

                try:
                    opus_device = resolve_device(options.hardware_mode, hardware_info)
                    opus_precision = resolve_precision(options.hardware_mode, hardware_info)
                    opus_batch = int(
                        options.translation_batch_size
                        or hardware_info.recommended_translation_batch
                    )
                    st.caption(
                        f"OPUS execution plan: **{opus_device.upper()} · "
                        f"{opus_precision} · batch {opus_batch}**"
                    )
                except Exception as exc:
                    st.warning(str(exc))
            else:
                st.warning("Local OPUS model has not been downloaded yet.")
                st.code(
                    f"pip install -r requirements-translation.txt\n"
                    f"python scripts/download_translation_models.py --pair "
                    f"{source_language}-{target_language}",
                    language="powershell",
                )

        elif translation_engine == TranslationEngine.OLLAMA:
            st.info(
                "**Recommended for high-quality EN→DE audiobook translation:** "
                "`translategemma:12b`. It is a dedicated translation model; "
                "the studio adds strict fidelity + natural Hochdeutsch rules."
            )
            if target_language == "de":
                address_options = ["du", "Sie"]
                current_address = (
                    options.translation_german_address_style
                    if options.translation_german_address_style in address_options
                    else "du"
                )
                options.translation_german_address_style = st.selectbox(
                    "German second-person style",
                    address_options,
                    index=address_options.index(current_address),
                    format_func=lambda value: (
                        "du — natural/direct audiobook style"
                        if value == "du"
                        else "Sie — formal address"
                    ),
                )

            if ollama_available():
                models = list_models()
                if models:
                    preferred = next(
                        (name for name in models if name.startswith("translategemma:12b")),
                        None,
                    )
                    current = (
                        options.translation_ollama_model
                        if options.translation_ollama_model in models
                        else (preferred or models[0])
                    )
                    options.translation_ollama_model = st.selectbox(
                        "Local Ollama translation model",
                        models,
                        index=models.index(current),
                    )
                    if options.translation_ollama_model.startswith("translategemma:12b"):
                        st.success(
                            "TranslateGemma 12B selected — literary translation preset + "
                            "chunked fidelity guard active. Long sections are translated in "
                            "smaller passages; summaries/English/truncated outputs are rejected automatically."
                        )
                    elif "translategemma" not in options.translation_ollama_model.lower():
                        st.caption(
                            "Generic Ollama model selected. The studio will use its "
                            "literary Hochdeutsch prompt and disable thinking for translation."
                        )
                else:
                    st.warning("Ollama is running but no local model is installed.")
                    st.code("ollama pull translategemma:12b", language="powershell")
            else:
                st.warning("Ollama is not running at 127.0.0.1:11434.")
                st.code(
                    "# After installing Ollama:\nollama pull translategemma:12b",
                    language="powershell",
                )

        else:
            st.info(
                "Argos is an optional lightweight fully-offline fallback. "
                "Install Argos plus its en↔de language package."
            )
            st.code(
                "pip install argostranslate\nargospm install translate-en_de",
                language="powershell",
            )

        project.save_translation_settings(
            {
                "enabled": options.translation_enabled,
                "source_language": source_language,
                "target_language": target_language,
                "engine": translation_engine.value,
                "model": options.translation_model,
                "ollama_model": options.translation_ollama_model,
                "german_address_style": options.translation_german_address_style,
            }
        )
        settings = project.load_settings()
        settings.update(options.to_dict())
        settings["selected_section_ids"] = st.session_state.selected_section_ids
        project.save_settings(settings)

        st.divider()
        st.subheader("Translation glossary")
        st.caption(
            "The glossary is not the translator. It is a consistency layer for recurring "
            "names and terms across the full book."
        )

        glossary_rows = [
            {"Source term": source, "Target term": target}
            for source, target in project.load_translation_glossary().items()
        ]
        edited_glossary = st.data_editor(
            glossary_rows,
            num_rows="dynamic",
            use_container_width=True,
            key=f"glossary_{project_key}",
        )

        if st.button("Save glossary", use_container_width=True):
            glossary = {}
            for row in as_records(edited_glossary):
                source = str(row.get("Source term", "")).strip()
                target = str(row.get("Target term", "")).strip()
                if source and target:
                    glossary[source] = target
            project.save_translation_glossary(glossary)
            st.success(f"Saved {len(glossary)} glossary entries.")

        selected_sections = selected_sections_from_state(sections)
        if not selected_sections:
            st.warning("Select at least one section in Blocks & Edit.")
        else:
            rows = translation_status(project, selected_sections, options)
            current_count = sum(1 for row in rows if row["current"])
            st.metric("Current translated sections", f"{current_count} / {len(rows)}")
            st.dataframe(
                [
                    {
                        "Chapter": row["chapter"],
                        "Section": row["title"],
                        "Ready": row["current"],
                        "Manual edit": row["manual"],
                    }
                    for row in rows
                ],
                hide_index=True,
                use_container_width=True,
            )

            tr1, tr2 = st.columns(2)

            with tr1:
                if st.button(
                    "Translate missing / changed sections",
                    type="primary",
                    use_container_width=True,
                    disabled=not translation_enabled,
                ):
                    progress = st.progress(0.0)
                    label = st.empty()

                    def translation_progress(message, fraction):
                        label.write(message)
                        progress.progress(max(0.0, min(1.0, float(fraction))))

                    try:
                        translate_sections_batch(
                            project,
                            selected_sections,
                            options,
                            force=False,
                            progress=translation_progress,
                        )
                        st.success("Translation cache updated.")
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)

            with tr2:
                if st.button(
                    "Force retranslate selected sections",
                    use_container_width=True,
                    disabled=not translation_enabled,
                ):
                    progress = st.progress(0.0)
                    label = st.empty()
                    try:
                        translate_sections_batch(
                            project,
                            selected_sections,
                            options,
                            force=True,
                            progress=lambda message, fraction: (
                                label.write(message),
                                progress.progress(max(0.0, min(1.0, float(fraction)))),
                            ),
                        )
                        st.success("Translations regenerated.")
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)

            st.divider()
            st.subheader("Original ↔ translation review")

            inspect_section = st.selectbox(
                "Section",
                selected_sections,
                format_func=lambda section: (
                    f"{section.chapter_title or section.title} → {section.title}"
                ),
                key="translation_inspect_section",
            )

            source_options = BuildOptions.from_dict(options.to_dict())
            source_options.pronunciations = {}
            source_prepared = humanize_sections(
                [inspect_section],
                source_options,
                project.load_overrides(),
                apply_pronunciation_rules=False,
            )
            source_text = (
                source_prepared[0][1] if source_prepared else inspect_section.raw_text
            )
            record = project.get_translation(inspect_section.id) or {}
            translated_text = str(
                record.get("manual_text", "").strip()
                or record.get("translated_text", "").strip()
            )

            original_col, translated_col = st.columns(2)
            with original_col:
                st.text_area(
                    "Cleaned source",
                    source_text[:30000],
                    height=440,
                    disabled=True,
                )
            with translated_col:
                edited_translation = st.text_area(
                    "Editable translation",
                    translated_text[:30000],
                    height=440,
                )

            rv1, rv2, rv3 = st.columns(3)

            with rv1:
                if st.button(
                    "Save manual translation edit",
                    use_container_width=True,
                    disabled=not bool(record),
                ):
                    try:
                        project.save_translation_manual_text(
                            inspect_section.id,
                            edited_translation,
                        )
                        st.success("Manual translation saved.")
                    except Exception as exc:
                        st.exception(exc)

            with rv2:
                if translated_text and tts_is_ready(options) and st.button(
                    "Test translated narration with final voice",
                    use_container_width=True,
                ):
                    try:
                        callback, preview_bar, preview_label = make_live_progress_ui()
                        path = generate_test_audio(
                            project.project_dir,
                            inspect_section,
                            edited_translation or translated_text,
                            options,
                            progress=callback,
                        )
                        preview_bar.progress(1.0)
                        preview_label.success("Translated test audio ready.")
                        st.session_state.translation_test_audio = str(path)
                    except Exception as exc:
                        st.exception(exc)

            with rv3:
                if translated_text and ollama_available() and list_models() and st.button(
                    "Local AI translation review",
                    use_container_width=True,
                ):
                    model = (
                        options.translation_ollama_model
                        if options.translation_ollama_model in list_models()
                        else list_models()[0]
                    )
                    try:
                        st.session_state.translation_review = review_translation(
                            source_text,
                            edited_translation or translated_text,
                            model,
                            source_language="English" if source_language == "en" else "German",
                            target_language="German" if target_language == "de" else "English",
                        )
                        st.session_state.translation_review_section = inspect_section.id
                    except Exception as exc:
                        st.exception(exc)

            translation_audio = st.session_state.get("translation_test_audio")
            if translation_audio and Path(translation_audio).exists():
                st.audio(Path(translation_audio).read_bytes(), format="audio/wav")

            proposal = st.session_state.get("translation_review", "")
            if (
                proposal
                and st.session_state.get("translation_review_section")
                == inspect_section.id
            ):
                reviewed = st.text_area(
                    "Local AI proposed translation — review before accepting",
                    proposal,
                    height=330,
                )
                if st.button("Accept reviewed translation", use_container_width=True):
                    project.save_translation_manual_text(inspect_section.id, reviewed)
                    st.success("Reviewed translation saved.")


with tab_language:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    st.subheader("Book-specific pronunciation dictionary")
    st.caption(
        "These rules apply to the final narration text. With translation enabled, "
        "they apply to the translated text."
    )

    rows = [
        {"Term": term, "Spoken as": spoken}
        for term, spoken in project.load_pronunciations().items()
    ]
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key=f"pron_{project_key}",
    )

    if st.button("Save pronunciation dictionary", use_container_width=True):
        clean = {}
        for row in as_records(edited):
            term = str(row.get("Term", "")).strip()
            spoken = str(row.get("Spoken as", "")).strip()
            if term and spoken:
                clean[term] = spoken
        project.save_pronunciations(clean)
        st.success(f"Saved {len(clean)} pronunciation rules.")

    if sections:
        selected = selected_sections_from_state(sections) or sections
        try:
            if options and options.translation_enabled:
                prepared = narration_sections(
                    project.project_dir,
                    selected,
                    options,
                    project.load_overrides(),
                )
                source_text = "\n".join(text for _, text in prepared)
            else:
                source_text = "\n".join(section.raw_text for section in selected)
        except Exception:
            source_text = "\n".join(section.raw_text for section in selected)

        suspects = find_suspicious_words(source_text)
        with st.expander("Suspicious names / acronyms to review", expanded=False):
            st.dataframe(
                [{"Word": word, "Occurrences": count} for word, count in suspects],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Pronunciation test")
    pt1, pt2 = st.columns(2)
    with pt1:
        test_term = st.text_input("Original term", value="Machiavelli")
    with pt2:
        test_spoken = st.text_input(
            "Proposed spoken form",
            value="mock-ee-ah-vell-ee",
        )

    if options and tts_is_ready(options) and st.button(
        "Generate original vs corrected pronunciation"
    ):
        options.pronunciations = project.load_pronunciations()
        try:
            original_audio, corrected_audio = generate_pronunciation_test(
                project.project_dir,
                test_term,
                test_spoken,
                options,
            )
            a, b = st.columns(2)
            with a:
                st.caption("Original")
                st.audio(original_audio.read_bytes(), format="audio/wav")
            with b:
                st.caption("Corrected")
                st.audio(corrected_audio.read_bytes(), format="audio/wav")
        except Exception as exc:
            st.exception(exc)

    st.subheader("Voice audition")
    if sections and options and tts_is_ready(options):
        audition_section = st.selectbox(
            "Audition passage",
            sections,
            format_func=lambda section: section.title,
            key="audition_section",
        )

        if options.translation_enabled:
            try:
                prepared = narration_sections(
                    project.project_dir,
                    [audition_section],
                    options,
                    project.load_overrides(),
                )
            except Exception:
                prepared = []
        else:
            prepared = humanize_sections(
                [audition_section],
                options,
                project.load_overrides(),
            )

        passage = prepared[0][1] if prepared else audition_section.raw_text

        if options.tts_engine == TTSEngine.KOKORO:
            voice_names = st.multiselect(
                "Compare voices",
                list(VOICE_PRESETS.keys()),
                default=[
                    "American — Heart",
                    "American — Sarah",
                    "British — George",
                ],
            )

            if st.button("Generate voice comparison"):
                st.session_state.voice_auditions = []
                for name in voice_names:
                    path = generate_test_audio(
                        project.project_dir,
                        audition_section,
                        passage,
                        options,
                        voice=VOICE_PRESETS[name],
                    )
                    st.session_state.voice_auditions.append((name, str(path)))

            for name, path in st.session_state.get("voice_auditions", []):
                if Path(path).exists():
                    st.markdown(f"**{name}**")
                    st.audio(Path(path).read_bytes(), format="audio/wav")

        elif options.tts_engine == TTSEngine.CHATTERBOX:
            st.caption(
                "Chatterbox can use its built-in voice or a local same-language reference "
                "voice. For Hochdeutsch, the reference clip is usually the more useful test."
            )

            if st.button(
                "Generate Chatterbox built-in/reference comparison",
                use_container_width=True,
            ):
                st.session_state.voice_auditions = []

                built_in_options = BuildOptions.from_dict(options.to_dict())
                built_in_options.chatterbox_reference_path = ""
                built_in_path = generate_test_audio(
                    project.project_dir,
                    audition_section,
                    passage,
                    built_in_options,
                )
                st.session_state.voice_auditions.append(
                    ("Chatterbox built-in voice", str(built_in_path))
                )

                if (
                    options.chatterbox_reference_path
                    and Path(options.chatterbox_reference_path).exists()
                ):
                    reference_options = BuildOptions.from_dict(options.to_dict())
                    reference_path = generate_test_audio(
                        project.project_dir,
                        audition_section,
                        passage,
                        reference_options,
                    )
                    st.session_state.voice_auditions.append(
                        ("Chatterbox local reference voice", str(reference_path))
                    )

            for name, path in st.session_state.get("voice_auditions", []):
                if Path(path).exists():
                    st.markdown(f"**{name}**")
                    st.audio(Path(path).read_bytes(), format="audio/wav")

        else:
            st.caption(
                "For Qwen3-TTS, change speaker/style in Setup and generate the same test passage."
            )

    st.divider()
    st.subheader("Narration replacement rules")

    rules_rows = [
        {
            "Enabled": bool(rule.get("enabled", True)),
            "Find": rule.get("find", ""),
            "Replace": rule.get("replace", ""),
            "Regex": bool(rule.get("regex", False)),
        }
        for rule in project.load_replacement_rules()
    ]
    edited_rules = st.data_editor(
        rules_rows,
        num_rows="dynamic",
        use_container_width=True,
        key=f"rules_{project_key}",
    )

    if st.button("Save replacement rules", use_container_width=True):
        saved = []
        for row in as_records(edited_rules):
            find = str(row.get("Find", "")).strip()
            if find:
                saved.append(
                    {
                        "enabled": bool(row.get("Enabled", True)),
                        "find": find,
                        "replace": str(row.get("Replace", "")),
                        "regex": bool(row.get("Regex", False)),
                    }
                )
        project.save_replacement_rules(saved)
        st.success(f"Saved {len(saved)} replacement rule(s).")

    st.subheader("Never narrate these exact strings")
    skip_value = st.text_area(
        "One string per line",
        value="\n".join(project.load_skip_strings()),
        height=130,
    )
    if st.button("Save skip-everywhere rules"):
        project.save_skip_strings(
            [value.strip() for value in skip_value.splitlines() if value.strip()]
        )
        st.success("Skip rules saved.")


with tab_queue:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    if not sections or options is None:
        st.info("Analyze the book and save setup first.")
    else:
        options.pronunciations = project.load_pronunciations()
        options.replacement_rules = project.load_replacement_rules()
        options.skip_strings = project.load_skip_strings()
        selected_sections = selected_sections_from_state(sections)

        translations_ready = True
        if options.translation_enabled:
            status_rows = translation_status(project, selected_sections, options)
            missing = [row for row in status_rows if not row["current"]]
            translations_ready = not missing
            if missing:
                st.error(
                    f"{len(missing)} selected section(s) need current local translations. "
                    "Finish the Translation tab first."
                )

        ready = bool(selected_sections) and translations_ready and tts_is_ready(options)
        queue_stats = None

        if ready:
            try:
                stats = estimate_stats(
                    selected_sections,
                    options,
                    project.load_overrides(),
                    project.project_dir,
                )
                queue_stats = stats
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Narration words", f"{stats['words']:,}")
                m2.metric(
                    "Estimated audiobook",
                    fmt_seconds(stats["audiobook_seconds"]),
                )

                backend_name = str(stats.get("inference_backend", "") or "")
                backend_display = {
                    "cuda_graph": "Fast Quality",
                    "official": "Official",
                    "lean": "Lean Eager",
                }.get(backend_name, backend_name or "Current TTS")
                speed_x = stats.get("times_realtime")
                m3.metric(
                    "TTS mode",
                    backend_display,
                    (
                        f"{float(speed_x):.2f}× realtime"
                        if speed_x
                        else None
                    ),
                )
                m4.metric(
                    "Estimated TTS time",
                    fmt_seconds(stats["generation_seconds"]),
                )
                st.caption(
                    "ETA source: "
                    f"{stats.get('estimate_source', 'uncalibrated')}. "
                    "It automatically switches to recent real chapter measurements "
                    "as the queue produces audio."
                )
            except Exception as exc:
                st.warning(str(exc))

        st.subheader("Persistent chapter queue")
        render_live_queue_status(str(project.project_dir))
        worker = worker_status(project)

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            if st.button(
                "Create / reset queue",
                use_container_width=True,
                disabled=worker.get("active", False) or not ready,
            ):
                jobs = []
                selected_set = {section.id for section in selected_sections}
                chapter_index = 0

                for _, title, members in chapter_groups(sections):
                    members = [
                        section for section in members if section.id in selected_set
                    ]
                    if not members:
                        continue

                    chapter_index += 1

                    chapter_words = 0
                    try:
                        chapter_prepared = narration_sections(
                            project.project_dir,
                            members,
                            options,
                            project.load_overrides(),
                        )
                        chapter_words = sum(
                            word_count(text)
                            for _, text in chapter_prepared
                        )
                    except Exception:
                        chapter_words = 0

                    estimated_seconds = None
                    estimated_audio_seconds = None
                    if (
                        queue_stats
                        and queue_stats.get("words")
                        and chapter_words > 0
                    ):
                        ratio = chapter_words / max(
                            1,
                            int(queue_stats["words"]),
                        )
                        if queue_stats.get("generation_seconds") is not None:
                            estimated_seconds = (
                                float(queue_stats["generation_seconds"]) * ratio
                            )
                        estimated_audio_seconds = (
                            float(queue_stats["audiobook_seconds"]) * ratio
                        )

                    jobs.append(
                        {
                            "title": title,
                            "output_name": f"{chapter_index:02d}_{title}",
                            "section_ids": [section.id for section in members],
                            "options": options.to_dict(),
                            "estimated_seconds": estimated_seconds,
                            "estimated_audio_seconds": estimated_audio_seconds,
                            "estimated_words": chapter_words,
                            "performance_key": (
                                queue_stats.get("performance_key")
                                if queue_stats
                                else ""
                            ),
                        }
                    )

                set_jobs(project, jobs)
                set_control(project, paused=True, stop_after_current=False)
                st.success(f"Created {len(jobs)} chapter job(s).")

        with q2:
            if st.button(
                "Start / Resume",
                type="primary",
                use_container_width=True,
                disabled=not ready,
            ):
                set_control(project, paused=False, stop_after_current=False)
                pid = start_worker(project)
                st.success(f"Queue worker started/resumed (PID {pid}).")

        with q3:
            if st.button("Pause", use_container_width=True):
                set_control(project, paused=True)
                st.info("The worker pauses after the current synthesis call.")

        with q4:
            if st.button("Stop after current chapter", use_container_width=True):
                set_control(project, stop_after_current=True)
                st.info("The worker will pause after the current chapter.")

        queue = get_queue(project)
        jobs = queue.get("jobs", [])

        if jobs:
            st.dataframe(
                [
                    {
                        "#": index + 1,
                        "Chapter": job.get("title"),
                        "Status": job.get("status"),
                        "Error": job.get("error", ""),
                    }
                    for index, job in enumerate(jobs)
                ],
                use_container_width=True,
                hide_index=True,
            )

            r1, r2, r3 = st.columns(3)
            with r1:
                if st.button("Refresh queue status", use_container_width=True):
                    st.rerun()
            with r2:
                if st.button("Retry failed jobs", use_container_width=True):
                    data = get_queue(project)
                    for job in data.get("jobs", []):
                        if job.get("status") == "failed":
                            job["status"] = "retry"
                            job["error"] = ""
                    project.save_json(project.queue_file, data)
                    st.success("Failed jobs marked for retry.")
            with r3:
                if st.button(
                    "Remove completed/failed jobs",
                    use_container_width=True,
                    disabled=worker.get("active", False),
                ):
                    clear_finished(project)
                    st.success("Finished queue rows removed; audio remains.")
        else:
            st.info("Queue is empty.")

        with st.expander("Generate selected material immediately (foreground)"):
            if st.button("Generate now without queue", disabled=not ready):
                progress = st.progress(0.0)
                label = st.empty()
                try:
                    outputs = build_audio(
                        project.project_dir,
                        selected_sections,
                        options,
                        overrides=project.load_overrides(),
                        progress=lambda message, fraction: (
                            label.write(message),
                            progress.progress(
                                max(0.0, min(1.0, float(fraction)))
                            ),
                        ),
                    )
                    st.success(f"Generated {len(outputs)} output file(s).")
                except Exception as exc:
                    st.exception(exc)


with tab_chunks:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    if not sections or options is None or not tts_is_ready(options):
        st.info("Analyze the book and start/install the selected local TTS engine.")
    else:
        options.pronunciations = project.load_pronunciations()
        options.replacement_rules = project.load_replacement_rules()
        options.skip_strings = project.load_skip_strings()
        selected_sections = selected_sections_from_state(sections)

        try:
            with st.spinner("Building chunk index..."):
                rows = inspect_chunks(
                    project.project_dir,
                    selected_sections,
                    options,
                    project.load_overrides(),
                )
        except Exception as exc:
            st.warning(str(exc))
            rows = []

        if rows:
            st.dataframe(
                [
                    {
                        "Chunk": row["chunk_id"],
                        "Chapter": row["chapter"],
                        "Section": row["section"],
                        "Words": row["words"],
                        "Source": row["pages"],
                        "Cached": row["cached"],
                        "Duration": round(row["duration"], 1),
                    }
                    for row in rows
                ],
                use_container_width=True,
                hide_index=True,
            )

            row_map = {row["chunk_id"]: row for row in rows}
            chunk_id = st.selectbox("Inspect chunk", list(row_map.keys()))
            row = row_map[chunk_id]

            st.text_area("Narration text", row["text"], height=260, disabled=True)

            if row["cached"] and Path(row["audio_path"]).exists():
                st.audio(Path(row["audio_path"]).read_bytes(), format="audio/wav")

            st.subheader("Report a problem")
            issue_type = st.selectbox(
                "Issue type",
                [
                    "Bad pronunciation",
                    "Wrong cleanup",
                    "Bad translation",
                    "Strange pause",
                    "Repeated text",
                    "Missing text",
                    "Other",
                ],
            )
            issue_note = st.text_input("Note")

            if st.button("Save issue for this chunk"):
                project.add_issue(
                    {
                        "chunk_id": chunk_id,
                        "section_id": row["section_id"],
                        "type": issue_type,
                        "note": issue_note,
                    }
                )
                st.success("Issue saved.")


with tab_export:
    st.subheader("Book metadata")
    meta = project.load_meta()

    mc1, mc2 = st.columns(2)
    with mc1:
        book_title = st.text_input(
            "Title",
            value=str(meta.get("title", Path(uploaded.name).stem)),
        )
    with mc2:
        author = st.text_input("Author", value=str(meta.get("author", "")))

    cover_page = 1
    if kind == "PDF":
        cover_page = st.number_input(
            "Cover PDF page",
            1,
            source_count,
            int(meta.get("cover_page", 1)),
            1,
        )

    if st.button("Save metadata" + (" & render cover" if kind == "PDF" else "")):
        project.update_meta(
            title=book_title,
            author=author,
            cover_page=int(cover_page),
        )
        if kind == "PDF":
            try:
                project.render_cover(int(cover_page), force=True)
            except Exception as exc:
                st.warning(f"Metadata saved, cover render failed: {exc}")
        st.success("Metadata saved.")

    if project.cover_file.exists():
        st.image(str(project.cover_file), width=220, caption="Audiobook cover")

    st.subheader("Finished audio")
    output_files = sorted(
        [
            path
            for path in project.output_dir.glob("*")
            if path.suffix.lower() in {".mp3", ".wav", ".m4b"}
        ],
        key=lambda path: path.name.lower(),
    )

    if output_files:
        output = st.selectbox("Audio file", output_files, format_func=lambda path: path.name)
        mime = (
            "audio/mp4"
            if output.suffix.lower() == ".m4b"
            else ("audio/mpeg" if output.suffix.lower() == ".mp3" else "audio/wav")
        )
        st.audio(output.read_bytes(), format=mime)
        st.download_button(
            "Download selected audio",
            output.read_bytes(),
            file_name=output.name,
            mime=mime,
        )

        progress_data = project.load_json(project.progress_file, {}) or {}
        current = progress_data.get(output.name, 0.0)
        timestamp = st.text_input(
            "Position (HH:MM:SS or MM:SS)",
            value=fmt_seconds(float(current)),
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("Save listening position"):
                project.set_listening_progress(output.name, parse_timestamp(timestamp))
                st.success("Position saved.")

        with pc2:
            bookmark_label = st.text_input("Bookmark label", value="Important point")
            if st.button("Add bookmark"):
                project.add_bookmark(
                    output.name,
                    parse_timestamp(timestamp),
                    bookmark_label,
                )
                st.success("Bookmark saved.")
    else:
        st.info("No finished output files yet.")

    bookmarks = project.load_json(project.bookmarks_file, []) or []
    if bookmarks:
        with st.expander(f"Bookmarks ({len(bookmarks)})"):
            st.dataframe(bookmarks, use_container_width=True, hide_index=True)

    st.subheader("Create one chapterized M4B")
    st.caption(
        "The exporter streams chapter files directly into the final M4B, avoiding the "
        "old giant temporary WAV. Export progress is shown below."
    )
    if st.button("Export M4B audiobook"):
        progress_bar = st.progress(0, text="Preparing M4B export...")
        progress_text = st.empty()

        def _m4b_progress(label: str, fraction: float) -> None:
            value = max(0.0, min(1.0, float(fraction)))
            progress_bar.progress(value, text=label)
            progress_text.caption(f"{value * 100:.0f}% — {label}")

        try:
            meta = project.load_meta()
            cover = project.cover_file if project.cover_file.exists() else None
            path = export_project_m4b(
                project.project_dir,
                title=str(meta.get("title", "Audiobook")),
                author=str(meta.get("author", "")),
                cover_path=cover,
                progress=_m4b_progress,
            )
            progress_bar.progress(1.0, text="M4B complete")
            progress_text.caption("100% — M4B complete")
            st.success(f"Created {path.name}")
            st.session_state.m4b_path = str(path)
        except Exception as exc:
            progress_text.empty()
            st.exception(exc)

    m4b = st.session_state.get("m4b_path")
    if m4b and Path(m4b).exists():
        st.download_button(
            "Download M4B",
            Path(m4b).read_bytes(),
            file_name=Path(m4b).name,
            mime="audio/mp4",
        )


with tab_ai:
    sections = st.session_state.get("sections", [])
    options = st.session_state.get("current_options")

    st.subheader("Optional local AI narration review")
    st.caption(
        "Ollama is localhost-only. Nothing is accepted automatically; "
        "you review every proposal."
    )

    if not ollama_available():
        st.info("Ollama not detected at 127.0.0.1:11434.")
    elif not sections or options is None:
        st.info("Analyze the book first.")
    else:
        models = list_models()

        if not models:
            st.warning("Ollama is running but no local model is installed.")
        else:
            model = st.selectbox("Local model", models)
            section = st.selectbox(
                "Section to review",
                sections,
                format_func=lambda item: item.title,
                key="ai_section",
            )

            try:
                prepared = narration_sections(
                    project.project_dir,
                    [section],
                    options,
                    project.load_overrides(),
                )
            except Exception:
                prepared = humanize_sections(
                    [section],
                    options,
                    project.load_overrides(),
                )

            current_text = prepared[0][1] if prepared else section.raw_text
            st.text_area(
                "Current narration",
                current_text[:20000],
                height=260,
                disabled=True,
            )

            if st.button("Ask local AI for a narration proposal"):
                with st.spinner("Local model is reviewing the passage..."):
                    try:
                        st.session_state.ai_proposal = propose_spoken_rewrite(
                            current_text,
                            model,
                        )
                        st.session_state.ai_section_id = section.id
                    except Exception as exc:
                        st.exception(exc)

            proposal = st.session_state.get("ai_proposal", "")
            if proposal and st.session_state.get("ai_section_id") == section.id:
                proposed_text = st.text_area(
                    "Proposal — review before accepting",
                    proposal,
                    height=300,
                )
                if st.button("Accept as narration override"):
                    if options.translation_enabled:
                        st.warning(
                            "For translated books, edit the final target text in the "
                            "Translation tab instead."
                        )
                    else:
                        project.save_override(section.id, proposed_text)
                        st.success("Accepted as section override.")


with tab_project:
    st.subheader("Privacy")
    st.success(
        "Runtime Privacy Lock active: loopback-only sockets, local Streamlit binding, "
        "model-hub offline mode, and telemetry disabled."
    )
    st.info(
        "The explicit download scripts are the only intended internet step. "
        "They download model weights; they do not upload your books."
    )

    st.subheader("Hardware & acceleration")

    hw1, hw2, hw3 = st.columns(3)

    with hw1:
        st.markdown("**Main app / OPUS**")
        st.write(f"CPU: {hardware_info.cpu_name}")
        st.write(f"Logical threads: {hardware_info.logical_cpus}")
        if hardware_info.gpu_name:
            st.write(f"GPU: {hardware_info.gpu_name}")
            if hardware_info.gpu_vram_gb:
                st.write(f"VRAM: {hardware_info.gpu_vram_gb:.1f} GB")
        else:
            st.write("GPU: not detected")

        st.write(
            "Torch CUDA: "
            + ("ready" if hardware_info.cuda_available else "not active")
        )
        st.write(
            f"Auto precision: {hardware_info.recommended_precision}"
        )
        st.write(
            f"Auto OPUS batch: {hardware_info.recommended_translation_batch}"
        )

    with hw2:
        st.markdown("**Chatterbox V3**")
        cb_health = chatterbox_server_health()
        if cb_health:
            st.success("Running")
            st.write(f"Device: {cb_health.get('device', '-')}")
            if cb_health.get("gpu_name"):
                st.write(f"GPU: {cb_health.get('gpu_name')}")
            if cb_health.get("vram_gb"):
                st.write(f"VRAM: {cb_health.get('vram_gb')} GB")
            st.write(
                "Reference: "
                + (
                    "custom local"
                    if cb_health.get("custom_reference_active")
                    else "built-in"
                )
            )
        elif chatterbox_model_ready():
            st.info("Model installed; sidecar not running.")
        else:
            st.info("Not installed.")

    with hw3:
        st.markdown("**Qwen3-TTS · optional**")
        qwen_health = qwen_server_health()
        if qwen_health:
            st.success("Running")
            st.write(f"Device: {qwen_health.get('device', '-')}")
            st.write(f"Precision: {qwen_health.get('precision', '-')}")
            if qwen_health.get("gpu_name"):
                st.write(f"GPU: {qwen_health.get('gpu_name')}")
            if qwen_health.get("vram_gb"):
                st.write(f"VRAM: {qwen_health.get('vram_gb')} GB")
        else:
            st.info("Not running.")

    if hardware_info.gpu_name and not hardware_info.cuda_available:
        st.warning(
            "Your NVIDIA GPU is visible, but the main Python environment cannot use it yet. "
            "Run `Setup-GPU-Acceleration.bat`, then restart the app."
        )
    elif hardware_info.cuda_available:
        st.success(
            "Automatic acceleration is active. OPUS uses CUDA with the safest supported "
            "precision; CPU-only work uses the available CPU threads."
        )

    st.caption(
        "Normal narration uses Chatterbox V3. Qwen remains available only as "
        "an optional fallback, so unnecessary GPU sidecars can stay unloaded."
    )

    st.divider()

    st.subheader("Storage & cache cleaner")
    storage = project_storage_report(project.project_dir)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Project size", format_bytes(storage["total"]))
    s2.metric("TTS chunk cache", format_bytes(storage["tts_chunk_cache"]))
    s3.metric("Preview cache", format_bytes(storage["preview_cache"]))
    s4.metric("Intermediate WAV", format_bytes(storage["intermediate_wav"]))

    st.dataframe(
        [
            {"Category": "Source book", "Size": format_bytes(storage["source"]), "Safe to delete?": "No"},
            {"Category": "Translations / settings / metadata", "Size": format_bytes(storage["metadata_and_text"]), "Safe to delete?": "No"},
            {"Category": "TTS chunk cache", "Size": format_bytes(storage["tts_chunk_cache"]), "Safe to delete?": "Yes — regenerable"},
            {"Category": "Short preview/test cache", "Size": format_bytes(storage["preview_cache"]), "Safe to delete?": "Yes"},
            {"Category": "Intermediate WAV", "Size": format_bytes(storage["intermediate_wav"]), "Safe to delete?": "Only when matching MP3 exists"},
            {"Category": "Final MP3", "Size": format_bytes(storage["final_mp3"]), "Safe to delete?": "No — final output"},
            {"Category": "Final M4B", "Size": format_bytes(storage["final_m4b"]), "Safe to delete?": "No — final output"},
        ],
        hide_index=True,
        use_container_width=True,
    )

    if storage["total"] >= 5 * 1024**3:
        st.warning(
            "This project is larger than 5 GB. The TTS chunk cache and intermediate WAV files "
            "are usually the biggest removable items."
        )

    worker_now = worker_status(project)
    destructive_disabled = bool(worker_now.get("active"))

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(
            "Safe cleanup",
            use_container_width=True,
            disabled=destructive_disabled,
            help="Deletes previews, temp files, and WAV files that already have a matching MP3.",
        ):
            result = safe_cleanup_project(project.project_dir)
            st.success(
                f"Removed {result.files_removed} file(s), freeing {format_bytes(result.bytes_removed)}."
            )
            st.rerun()

    with c2:
        if st.button(
            "Clear short preview cache",
            use_container_width=True,
            disabled=destructive_disabled,
        ):
            result = clear_preview_cache(project.project_dir)
            st.success(
                f"Removed {result.files_removed} preview file(s), freeing {format_bytes(result.bytes_removed)}."
            )
            st.rerun()

    with c3:
        if st.button(
            "Remove redundant WAV files",
            use_container_width=True,
            disabled=destructive_disabled,
            help="Only removes chapter WAV files when a same-name MP3 already exists.",
        ):
            result = clear_redundant_intermediate_wavs(project.project_dir)
            st.success(
                f"Removed {result.files_removed} WAV file(s), freeing {format_bytes(result.bytes_removed)}."
            )
            st.rerun()

    st.markdown("**TTS cache management**")
    st.caption(
        "Chunk audio is fully regenerable. Keeping it makes future edits fast; deleting it saves disk space."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button(
            "Prune old TTS cache, keep current setup",
            use_container_width=True,
            disabled=destructive_disabled,
            help=(
                "Keeps cached chunks used by the currently selected sections/voice/settings and "
                "deletes old chunks left behind by previous voices, speeds, text edits, or engines."
            ),
        ):
            try:
                current_sections = st.session_state.get("sections", [])
                current_options = st.session_state.get("current_options")
                selected = selected_sections_from_state(current_sections)
                if not selected or current_options is None:
                    raise RuntimeError("Analyze and select the current book sections first.")

                current_options.pronunciations = project.load_pronunciations()
                current_options.replacement_rules = project.load_replacement_rules()
                current_options.skip_strings = project.load_skip_strings()

                rows = inspect_chunks(
                    project.project_dir,
                    selected,
                    current_options,
                    project.load_overrides(),
                )
                keep_paths = [
                    row["audio_path"]
                    for row in rows
                    if row.get("cached") and row.get("audio_path")
                ]
                result = prune_tts_chunk_cache(project.project_dir, keep_paths)
                st.success(
                    f"Pruned {result.files_removed} obsolete chunk(s), freeing "
                    f"{format_bytes(result.bytes_removed)}."
                )
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    with cc2:
        confirm_clear_chunks = st.checkbox(
            "I understand the chunk cache will need to be regenerated",
            key="confirm_clear_tts_cache",
        )
        if st.button(
            "Clear ALL TTS chunk cache",
            use_container_width=True,
            disabled=destructive_disabled or not confirm_clear_chunks,
        ):
            result = clear_tts_chunk_cache(project.project_dir)
            st.success(
                f"Removed {result.files_removed} chunk file(s), freeing {format_bytes(result.bytes_removed)}."
            )
            st.rerun()

    st.markdown("**Compact a finished project**")
    st.caption(
        "Keeps the source book, translations, glossary, edits, settings, final MP3/M4B files, "
        "and bookmarks. Deletes regenerable chunks, previews, temp files, and redundant WAVs."
    )
    confirm_compact = st.checkbox(
        "I have checked that my final MP3/M4B output is complete",
        key="confirm_compact_project",
    )
    if st.button(
        "Compact finished project",
        type="secondary",
        use_container_width=True,
        disabled=destructive_disabled or not confirm_compact,
    ):
        result = compact_finished_project(project.project_dir)
        st.success(
            f"Project compacted: removed {result.files_removed} file(s), "
            f"freeing {format_bytes(result.bytes_removed)}."
        )
        st.rerun()

    with st.expander("All audiobook projects on disk"):
        all_rows = all_projects_report(DATA_DIR)
        st.dataframe(
            [
                {
                    "Project": row["project"],
                    "Total": format_bytes(row["total"]),
                    "TTS cache": format_bytes(row["tts_chunk_cache"]),
                    "Previews": format_bytes(row["preview_cache"]),
                    "Intermediate WAV": format_bytes(row["intermediate_wav"]),
                    "Final audio": format_bytes(row["final_mp3"] + row["final_m4b"]),
                }
                for row in all_rows
            ],
            hide_index=True,
            use_container_width=True,
        )
        if st.button(
            "Global safe cleanup (all books)",
            disabled=destructive_disabled,
            help="Only removes previews, temp files, and redundant paired WAVs across all projects.",
        ):
            result = global_safe_cleanup(DATA_DIR)
            st.success(
                f"Global cleanup removed {result.files_removed} file(s), freeing "
                f"{format_bytes(result.bytes_removed)}."
            )
            st.rerun()

    st.divider()

    st.subheader("Persistent project")
    st.code(str(project.project_dir))
    st.write(
        "Future versions can replace the app code while preserving `.audiobook_data/` "
        "and `models/`."
    )

    st.dataframe(
        [
            {"File": "pronunciation.json", "Purpose": "book-specific final narration pronunciation"},
            {"File": "translation_glossary.json", "Purpose": "source → target term consistency"},
            {"File": "translation_memory.json", "Purpose": "reuse identical local translations"},
            {"File": "translations.json", "Purpose": "cached + manually edited translations"},
            {"File": "replacement_rules.json", "Purpose": "spoken-text transformations"},
            {"File": "narration_overrides.json", "Purpose": "manual cleaned-source overrides"},
            {"File": "analysis.json", "Purpose": "saved structure and semantic blocks"},
            {"File": "queue.json", "Purpose": "persistent generation queue"},
            {"File": "issues.json", "Purpose": "problem reports"},
            {"File": "bookmarks.json", "Purpose": "listening bookmarks"},
            {"File": "output_registry.json", "Purpose": "finished chapters for M4B"},
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Local model setup commands")
    st.markdown("**English Kokoro**")
    st.code("python scripts/download_models.py", language="powershell")

    st.markdown("**English → German OPUS translation**")
    st.code(
        "pip install -r requirements-translation.txt\n"
        "python scripts/download_translation_models.py --pair en-de",
        language="powershell",
    )

    st.markdown("**GPU acceleration for your NVIDIA GPU**")
    st.code(
        ".\\Setup-GPU-Acceleration.bat",
        language="powershell",
    )
    st.caption(
        "Checks/upgrades CUDA-enabled Torch in the main app and optional Qwen "
        "environment when present."
    )

    st.markdown("**Chatterbox Multilingual V3 — recommended natural German test**")
    st.code(
        ".\\Setup-Chatterbox-GPU.bat",
        language="powershell",
    )
    st.caption(
        "Creates `.venv-chatterbox`, installs its pinned CUDA runtime, downloads "
        "the multilingual V3 model, and runs a local German smoke test. "
        "Normal runtime uses `from_local()` only."
    )

    with st.expander("Optional Qwen3-TTS setup"):
        st.code(
            "py -3.12 -m venv .venv-qwen\n"
            ".venv-qwen\\Scripts\\Activate.ps1\n"
            "pip install -U qwen-tts soundfile huggingface_hub\n"
            "python scripts\\download_qwen3_tts.py --model 0.6b-custom\n"
            "python scripts\\qwen_tts_server.py "
            "--model-dir models\\qwen3-tts\\Qwen3-TTS-12Hz-0.6B-CustomVoice "
            "--mode custom_voice",
            language="powershell",
        )

    st.warning("Do not delete `.audiobook_data/` or `models/` when upgrading.")
