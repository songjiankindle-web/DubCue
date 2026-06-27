from voxcpm.longform import (
    build_director_segments,
    concat_with_pauses,
    expected_duration_range,
    final_text_for_segment,
    rows_to_segments,
    segments_to_rows,
    smart_split_text,
    validate_or_repair_segment_audio,
    natural_speed_report,
)

import numpy as np
from app import apply_director_keyboard_command, director_audio_path, split_director_row, merge_director_rows


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
    rows[0][3] = "同一位纪录片旁白，语速缓慢，情绪带有希望感"

    segments = rows_to_segments(rows)

    assert len(segments) == 1
    assert "希望" in segments[0].prompt
    assert segments[0].pause_ms >= 0
    assert len(rows[0]) == 7


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


def test_reference_speed_guard_warns_without_stretching_or_failing():
    sample_rate = 24000
    text = "这是一段需要慢速讲述的纪录片旁白。"
    fast_audio = np.sin(np.linspace(0, 120, int(sample_rate * 2.0))).astype(np.float32) * 0.05

    report = natural_speed_report(
        fast_audio,
        sample_rate,
        text,
        "slow",
        reference_cps=2.0,
    )

    assert report["too_fast"] is True
    assert len(fast_audio) == sample_rate * 2


def test_concat_enforces_one_second_pause_between_segments():
    sample_rate = 24000
    clip_a = np.ones(sample_rate, dtype=np.float32) * 0.1
    clip_b = np.ones(sample_rate, dtype=np.float32) * 0.1

    combined = concat_with_pauses([clip_a, clip_b], [120, 0], sample_rate)

    assert len(combined) == sample_rate * 3
    assert np.allclose(combined[sample_rate : sample_rate * 2], 0.0)


def test_director_row_split_invalidates_old_audio():
    rows = [[1, "前半句|后半句", "slow", "prompt", 1000, "done", "/tmp/old.wav"]]

    new_rows, selected, editor_text = split_director_row(rows, 0, "前半句|后半句")

    assert selected == 0
    assert editor_text == "前半句"
    assert [row[0] for row in new_rows] == [1, 2]
    assert [row[1] for row in new_rows] == ["前半句", "后半句"]
    assert new_rows[0][5] == "pending"
    assert director_audio_path(new_rows[0]) == ""
    assert director_audio_path(new_rows[1]) == ""


def test_director_row_merge_previous_invalidates_audio_and_reindexes():
    rows = [
        [1, "前半句", "slow", "prompt", 1000, "done", "/tmp/a.wav"],
        [2, "后半句", "slow", "prompt", 1000, "done", "/tmp/b.wav"],
    ]

    new_rows, selected, editor_text = merge_director_rows(rows, 1, "previous")

    assert selected == 0
    assert editor_text == "前半句后半句"
    assert len(new_rows) == 1
    assert new_rows[0][0] == 1
    assert new_rows[0][5] == "pending"
    assert director_audio_path(new_rows[0]) == ""


def test_director_keyboard_command_splits_and_merges_rows():
    rows = [[1, "前半句后半句", "slow", "prompt", 1000, "done", "/tmp/old.wav"]]

    rows, selected, _audio, _file, _status = apply_director_keyboard_command(
        rows,
        0,
        '{"action":"split","before":"前半句","after":"后半句"}',
    )

    assert selected == 0
    assert [row[1] for row in rows] == ["前半句", "后半句"]
    assert rows[0][5] == "pending"
    assert director_audio_path(rows[0]) == ""

    rows, selected, _audio, _file, _status = apply_director_keyboard_command(
        rows,
        1,
        '{"action":"merge_previous"}',
    )

    assert selected == 0
    assert len(rows) == 1
    assert rows[0][1] == "前半句后半句"
