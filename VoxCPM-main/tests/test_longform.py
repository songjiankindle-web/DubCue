from voxcpm.longform import (
    build_director_segments,
    rows_to_segments,
    segments_to_rows,
    smart_split_text,
)


def test_smart_split_prefers_sentence_boundaries():
    text = "雨停之后，村庄重新露出了轮廓。老人站在桥边，望着远处被洪水冲毁的田地。"

    segments = smart_split_text(text, max_chars=30)

    assert segments == [
        "雨停之后，村庄重新露出了轮廓。",
        "老人站在桥边，望着远处被洪水冲毁的田地。",
    ]


def test_director_rows_are_editable_roundtrip():
    text = "三个月后，第一批孩子回到了学校。操场上再次响起了笑声。"
    rows = segments_to_rows(build_director_segments(text, max_chars=30))
    rows[0][4] = "同一位纪录片旁白，语速缓慢，情绪带有希望感"

    segments = rows_to_segments(rows)

    assert len(segments) == 1
    assert "希望" in segments[0].prompt
    assert segments[0].pause_ms >= 0
