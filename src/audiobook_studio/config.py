from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = APP_ROOT / ".audiobook_data"
MODEL_DIR = APP_ROOT / "models"

KOKORO_MODEL = MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODEL_DIR / "voices-v1.0.bin"

DEFAULT_VOICE = "af_heart"
DEFAULT_LANGUAGE = "en-us"

VOICE_PRESETS = {
    "American — Heart": "af_heart",
    "American — Sarah": "af_sarah",
    "American — Nicole": "af_nicole",
    "American — Bella": "af_bella",
    "American — Michael": "am_michael",
    "American — Adam": "am_adam",
    "British — Emma": "bf_emma",
    "British — Isabella": "bf_isabella",
    "British — George": "bm_george",
    "British — Lewis": "bm_lewis",
}

KOKORO_LANGUAGES = {
    "American English": "en-us",
    "British English": "en-gb",
}

QWEN_LANGUAGES = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
}

# Qwen3-TTS CustomVoice built-in speaker names from the official release.
QWEN_SPEAKERS = [
    "Ryan",
    "Aiden",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
]



DEFAULT_PRONUNCIATIONS = {
    "AES": "A E S",
    "CVE": "C V E",
    "XSS": "X S S",
    "CSRF": "C S R F",
    "TLS": "T L S",
    "SSL": "S S L",
    "SSH": "S S H",
    "CPU": "C P U",
    "GPU": "G P U",
    "API": "A P I",
    "URL": "U R L",
    "HTTP": "H T T P",
    "HTTPS": "H T T P S",
    "SHA-256": "SHA two fifty six",
    "SHA-1": "SHA one",
    "SQL": "S Q L",
    "Nmap": "N map",
    "OAuth": "oh auth",
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


CHATTERBOX_MODEL_DIR = MODEL_DIR / "chatterbox" / "multilingual-v3"

CHATTERBOX_REQUIRED_FILES = (
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
)

CHATTERBOX_LANGUAGES = {
    "German": "de",
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Danish": "da",
    "Norwegian": "no",
    "Swedish": "sv",
    "Finnish": "fi",
    "Polish": "pl",
    "Russian": "ru",
    "Greek": "el",
    "Turkish": "tr",
    "Arabic": "ar",
    "Hebrew": "he",
    "Hindi": "hi",
    "Malay": "ms",
    "Swahili": "sw",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
}

CHATTERBOX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
