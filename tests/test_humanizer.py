from src.audiobook_studio.humanizer import humanize
from src.audiobook_studio.models import (
    HumanizerOptions,
    ReadingMode,
)


def natural(text: str):
    return humanize(
        text,
        HumanizerOptions(
            mode=ReadingMode.NATURAL
        ),
    )


def test_removes_numeric_citations():
    result = natural(
        "The attack is practical [12, 14, 19]."
    )
    assert "[12" not in result.text
    assert "The attack is practical" in result.text


def test_dehyphenates_pdf_line_wrap():
    result = natural(
        "The implemen-\ntation is secure."
    )
    assert "implementation" in result.text


def test_bullet_dash_is_not_spoken_as_minus():
    result = natural(
        "- confidentiality\n"
        "- integrity\n"
        "- availability"
    )

    assert "minus confidentiality" not in result.text
    assert "confidentiality" in result.text
    assert "availability" in result.text


def test_real_numeric_minus_is_spoken():
    result = natural(
        "The result is 10 - 4 = 6."
    )

    assert "10 minus 4" in result.text
    assert "equals" in result.text


def test_parentheses_are_naturalized():
    result = natural(
        "TLS (Transport Layer Security) "
        "protects the connection."
    )

    assert "(" not in result.text
    assert ")" not in result.text
    assert "Transport Layer Security" in result.text


def test_stops_at_references():
    result = natural(
        "Main chapter text.\n\n"
        "References\n"
        "[1] Example paper."
    )

    assert result.stopped_at_bibliography is True
    assert "Example paper" not in result.text


def test_numbered_subheading_is_spoken_naturally():
    result = natural(
        "4.2 Threat Model\n"
        "The adversary observes the system."
    )

    assert "4.2" not in result.text
    assert "Threat Model" in result.text
