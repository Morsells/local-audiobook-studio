from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chatterbox_runtime_uses_from_local_only():
    server = (
        ROOT / "scripts" / "chatterbox_tts_server.py"
    ).read_text(encoding="utf-8")

    assert "ChatterboxMultilingualTTS.from_local(" in server

    # Ignore the explanatory comment mentioning from_pretrained; there must be
    # no executable call to it in the runtime sidecar.
    executable_lines = [
        line
        for line in server.splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert ".from_pretrained(" not in "\n".join(executable_lines)


def test_chatterbox_launcher_and_setup_exist():
    assert (ROOT / "Setup-Chatterbox-GPU.bat").exists()
    assert (
        ROOT / "Start-Audiobook-Studio-With-Chatterbox.bat"
    ).exists()
    assert (
        ROOT / "scripts" / "download_chatterbox_models.py"
    ).exists()


def test_chatterbox_runtime_is_loopback_sidecar():
    server = (
        ROOT / "scripts" / "chatterbox_tts_server.py"
    ).read_text(encoding="utf-8")
    assert '("127.0.0.1", args.port)' in server
    assert "install_socket_guard()" in server
    assert "enforce_offline_model_runtime()" in server
