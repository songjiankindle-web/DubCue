from voxcpm.longform import (
    build_director_segments,
    expected_duration_range,
    final_text_for_segment,
    rows_to_segments,
    segments_to_rows,
    smart_split_text,
    validate_or_repair_segment_audio,
)

import numpy as np


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


def test_segment_prompt_is_not_spoken_by_default():
    segment = build_director_segments("中国国家博物馆内，C形碧玉龙格外醒目。")[0]

    assert final_text_for_segment(segment) == segment.text
    assert final_text_for_segment(segment, apply_prompt=True).startswith("(")


def test_long_silent_tail_is_capped():
    sample_rate = 24000
    segment = build_director_segments("很长时间里，这里一直默默无闻。")[0]
    _min_seconds, max_seconds = expected_duration_range(segment.text, segment.speed)
    speech = np.ones(int(sample_rate * 2.0), dtype=np.float32) * 0.03
    tail = np.zeros(int(sample_rate * 90.0), dtype=np.float32)

    repaired = validate_or_repair_segment_audio(
        np.concatenate([speech, tail]),
        sample_rate=sample_rate,
        segment=segment,
    )

    assert len(repaired) / sample_rate <= max_seconds * 1.08
