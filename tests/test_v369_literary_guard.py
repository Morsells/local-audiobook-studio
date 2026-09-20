from src.audiobook_studio.translation import _split_literary_translation_chunks, _translation_quality_issues

def test_chunk_split_preserves_word_count():
    source = ('word ' * 120 + '\n\n') * 12
    chunks = _split_literary_translation_chunks(source, max_words=300, max_chars=100000)
    assert len(chunks) >= 4
    assert sum(len(c.split()) for c in chunks) == len(source.split())

def test_guard_rejects_summary_marker_and_short_output():
    source = 'The power game requires patience and observation. ' * 50
    bad = "This is a fascinating excerpt. Here's a breakdown of the key themes. Core Themes: Power and Strategy."
    issues = _translation_quality_issues(source, bad, target_language='de')
    assert any('summary/commentary' in x for x in issues)
    assert any('suspiciously short' in x for x in issues)

def test_guard_rejects_predominantly_english_output():
    source = 'The power game requires patience and observation. ' * 20
    bad = 'The text discusses the power game and the importance of patience and observation in the court and in life.' * 3
    issues = _translation_quality_issues(source, bad, target_language='de')
    assert any('predominantly English' in x for x in issues)

def test_guard_accepts_normal_german_shape():
    source = 'Always make those above you feel comfortably superior. ' * 12
    good = 'Sorge stets dafür, dass sich diejenigen über dir wohl und überlegen fühlen. ' * 12
    assert _translation_quality_issues(source, good, target_language='de') == []
