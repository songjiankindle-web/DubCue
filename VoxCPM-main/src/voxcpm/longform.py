import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import soundfile as sf


SENTENCE_END_RE = re.compile(r"([。！？!?;；]+|(?:\.{3,}|…{1,2}))")
SOFT_BREAK_RE = re.compile(r"([，,、：:—-]+)")
SPACE_RE = re.compile(r"\s+")


@dataclass
class DirectorSegment:
    index: int
    text: str
    emotion: str
    speed: str
    prompt: str
    pause_ms: int
    status: str = "pending"


@dataclass
class LongformResult:
    output_path: str
    manifest_path: str
    segment_dir: str
    segment_count: int
    duration_seconds: float


class SegmentGenerationError(RuntimeError):
    pass


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def split_sentences(paragraph: str) -> list[str]:
    parts = SENTENCE_END_RE.split(paragraph)
    sentences: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if SENTENCE_END_RE.fullmatch(part):
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    pieces = SOFT_BREAK_RE.split(sentence)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = current + piece
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = piece
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars * 1.25:
            final.append(chunk)
            continue
        final.extend(_split_by_length_fallback(chunk, max_chars))
    return final


def _split_by_length_fallback(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            for offset in range(end, max(start + max_chars // 2, start), -1):
                if text[offset - 1].isspace():
                    end = offset
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def smart_split_text(text: str, max_chars: int = 70, allow_sentence_split: bool = True) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    segments: list[str] = []
    for paragraph in cleaned.split("\n"):
        sentences = split_sentences(paragraph)
        buffer = ""
        for sentence in sentences:
            sentence_parts = (
                split_long_sentence(sentence, max_chars)
                if allow_sentence_split and len(sentence) > max_chars * 1.4
                else [sentence]
            )
            for part in sentence_parts:
                if not buffer:
                    buffer = part
                    continue
                candidate = buffer + part
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    segments.append(buffer.strip())
                    buffer = part
        if buffer.strip():
            segments.append(buffer.strip())
    return segments


def analyze_emotion(text: str, base_style: str = "documentary") -> tuple[str, str, int]:
    lowered = text.lower()
    suspense_words = ("悬疑", "秘密", "黑暗", "阴影", "失踪", "谜", "危险", "沉默", "真相")
    sad_words = ("死亡", "离开", "废墟", "洪水", "战争", "眼泪", "孤独", "失去", "痛苦")
    hopeful_words = ("希望", "重新", "孩子", "笑声", "春天", "抵达", "光", "未来", "回到")
    intense_words = ("突然", "爆炸", "冲突", "追赶", "怒", "危机", "震惊", "breaking")
    calm_words = ("清晨", "雨停", "远处", "缓缓", "静静", "山谷", "海面", "夜色")

    if any(word in lowered or word in text for word in intense_words):
        return "紧张 / 推进", "standard", 320
    if any(word in text for word in sad_words):
        return "沉重 / 克制", "slow", 620
    if any(word in text for word in suspense_words):
        return "低沉 / 悬疑", "slow", 560
    if any(word in text for word in hopeful_words):
        return "温和 / 希望", "standard", 480
    if any(word in text for word in calm_words):
        return "安静 / 凝视", "slow", 560
    if base_style == "news":
        return "清晰 / 稳定", "standard", 360
    if base_style == "story":
        return "自然 / 叙述", "standard", 440
    return "克制 / 纪录片", "slow", 520


def build_control_prompt(
    *,
    emotion: str,
    speed: str,
    base_style: str,
    user_style: str = "",
    variation: str = "medium",
) -> str:
    speed_map = {
        "slow": "语速缓慢，句间留白自然",
        "standard": "语速平稳，咬字清晰",
        "fast": "语速略快但清楚，不抢话",
    }
    variation_map = {
        "low": "情绪变化克制，仅做轻微起伏",
        "medium": "情绪随文本自然起伏",
        "high": "情绪表达更明确，但不要夸张",
    }
    style_map = {
        "documentary": "同一位纪录片旁白",
        "news": "同一位新闻解说员",
        "story": "同一位故事讲述者",
        "commercial": "同一位商业广告旁白",
    }
    style = style_map.get(base_style, "同一位旁白")
    speed_text = speed_map.get(speed, speed_map["standard"])
    variation_text = variation_map.get(variation, variation_map["medium"])
    user_suffix = f"，整体风格：{user_style.strip()}" if user_style.strip() else ""
    return (
        f"{style}，保持相同音色和声音质感，只调整本段表达；"
        f"本段情绪：{emotion}；{speed_text}；{variation_text}"
        f"{user_suffix}"
    )


def build_director_segments(
    text: str,
    *,
    max_chars: int = 70,
    base_style: str = "documentary",
    user_style: str = "",
    variation: str = "medium",
    allow_sentence_split: bool = True,
) -> list[DirectorSegment]:
    chunks = smart_split_text(text, max_chars=max_chars, allow_sentence_split=allow_sentence_split)
    segments: list[DirectorSegment] = []
    for index, chunk in enumerate(chunks, 1):
        emotion, speed, pause_ms = analyze_emotion(chunk, base_style=base_style)
        prompt = build_control_prompt(
            emotion=emotion,
            speed=speed,
            base_style=base_style,
            user_style=user_style,
            variation=variation,
        )
        segments.append(
            DirectorSegment(
                index=index,
                text=chunk,
                emotion=emotion,
                speed=speed,
                prompt=prompt,
                pause_ms=pause_ms,
            )
        )
    return segments


def segments_to_rows(segments: Iterable[DirectorSegment]) -> list[list[object]]:
    return [
        [
            segment.index,
            segment.text,
            segment.emotion,
            segment.speed,
            segment.prompt,
            segment.pause_ms,
            segment.status,
        ]
        for segment in segments
    ]


def rows_to_segments(rows) -> list[DirectorSegment]:
    if rows is None:
        return []
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    segments: list[DirectorSegment] = []
    for i, row in enumerate(rows, 1):
        if not row or len(row) < 2:
            continue
        text = str(row[1] or "").strip()
        if not text:
            continue
        emotion = str(row[2] or "克制 / 纪录片").strip() if len(row) > 2 else "克制 / 纪录片"
        speed = str(row[3] or "slow").strip() if len(row) > 3 else "slow"
        prompt = str(row[4] or "").strip() if len(row) > 4 else ""
        pause_raw = row[5] if len(row) > 5 else 520
        try:
            pause_ms = int(float(pause_raw))
        except (TypeError, ValueError):
            pause_ms = 520
        if not prompt:
            prompt = build_control_prompt(
                emotion=emotion,
                speed=speed,
                base_style="documentary",
            )
        segments.append(
            DirectorSegment(
                index=i,
                text=text,
                emotion=emotion,
                speed=speed,
                prompt=prompt,
                pause_ms=max(0, min(pause_ms, 3000)),
                status="pending",
            )
        )
    return segments


def final_text_for_segment(segment: DirectorSegment, apply_prompt: bool = False) -> str:
    if not apply_prompt:
        return segment.text
    prompt = re.sub(r"[()（）]", "", segment.prompt).strip()
    return f"({prompt}){segment.text}" if prompt else segment.text


def chars_per_second(text: str, audio: np.ndarray, sample_rate: int) -> float:
    visible_chars = len(re.sub(r"\s+", "", text))
    duration = max(len(audio) / float(sample_rate), 0.001)
    return visible_chars / duration


def adjust_speed_if_needed(audio: np.ndarray, sample_rate: int, text: str, speed: str) -> np.ndarray:
    targets = {"slow": 2.8, "standard": 4.2, "fast": 5.6}
    target = targets.get(speed)
    if target is None:
        return audio
    actual = chars_per_second(text, audio, sample_rate)
    if actual <= target * 1.12:
        return audio
    rate = max(0.82, min(1.0, target / actual))
    try:
        import librosa

        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        return stretched.astype(np.float32)
    except Exception:
        return audio


def trim_edges(audio: np.ndarray, threshold: float = 0.003, keep_ms: int = 80, sample_rate: int = 24000) -> np.ndarray:
    if audio.size == 0:
        return audio
    mono = np.squeeze(audio).astype(np.float32)
    frame = max(256, int(sample_rate * 0.04))
    hop = max(128, frame // 2)
    if len(mono) < frame:
        return mono
    energies = []
    for start in range(0, len(mono) - frame + 1, hop):
        chunk = mono[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk))))
    if not energies:
        return mono
    energy = np.asarray(energies)
    adaptive = max(threshold, float(np.percentile(energy, 85)) * 0.08)
    voiced = np.flatnonzero(energy > adaptive)
    if voiced.size == 0:
        indices = np.flatnonzero(np.abs(mono) > threshold)
    else:
        start_frame = int(voiced[0])
        end_frame = int(voiced[-1])
        keep = int(sample_rate * keep_ms / 1000)
        start = max(0, start_frame * hop - keep)
        end = min(len(mono), end_frame * hop + frame + keep)
        return mono[start:end]
    if indices.size == 0:
        return mono
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, int(indices[0]) - keep)
    end = min(len(mono), int(indices[-1]) + keep)
    return mono[start:end]


def normalize_peak(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 0:
        return audio
    return (audio * min(1.0, peak / current)).astype(np.float32)


def concat_with_pauses(
    clips: list[np.ndarray],
    pauses_ms: list[int],
    sample_rate: int,
    crossfade_ms: int = 45,
) -> np.ndarray:
    if not clips:
        return np.zeros(0, dtype=np.float32)
    output = clips[0].astype(np.float32)
    fade_len = int(sample_rate * crossfade_ms / 1000)
    for clip, pause_ms in zip(clips[1:], pauses_ms[:-1]):
        pause = np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
        next_clip = clip.astype(np.float32)
        if fade_len > 0 and len(output) > fade_len and len(next_clip) > fade_len:
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            overlap = output[-fade_len:] * fade_out + next_clip[:fade_len] * fade_in
            output = np.concatenate([output[:-fade_len], overlap, next_clip[fade_len:], pause])
        else:
            output = np.concatenate([output, pause, next_clip])
    return normalize_peak(output)


def _to_numpy_audio(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    return np.squeeze(audio)


def expected_duration_range(text: str, speed: str) -> tuple[float, float]:
    chars = max(1, len(re.sub(r"\s+", "", text)))
    min_cps = {"slow": 1.35, "standard": 2.0, "fast": 2.8}.get(speed, 1.8)
    max_cps = {"slow": 4.2, "standard": 5.6, "fast": 7.0}.get(speed, 5.2)
    min_seconds = max(1.2, chars / max_cps - 1.0)
    max_seconds = min(35.0, max(8.0, chars / min_cps + 5.0))
    return min_seconds, max_seconds


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def validate_or_repair_segment_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    segment: DirectorSegment,
) -> np.ndarray:
    audio = trim_edges(audio, sample_rate=sample_rate)
    if audio.size == 0:
        raise SegmentGenerationError("Generated empty audio.")

    _min_seconds, max_seconds = expected_duration_range(segment.text, segment.speed)
    max_samples = int(max_seconds * sample_rate)
    duration = len(audio) / float(sample_rate)
    rms = audio_rms(audio)
    if rms < 0.0008:
        raise SegmentGenerationError(
            f"Generated audio is almost silent: rms={rms:.6f}, duration={duration:.2f}s."
        )
    if duration > max_seconds:
        # Most bad cases are valid speech followed by a very long low-energy tail.
        # Cap the segment instead of allowing a 30-character line to become minutes.
        audio = audio[:max_samples]
        audio = trim_edges(audio, sample_rate=sample_rate)
        duration = len(audio) / float(sample_rate)
        if duration > max_seconds * 1.08:
            raise SegmentGenerationError(
                f"Generated segment is too long: {duration:.2f}s > {max_seconds:.2f}s."
            )
    return audio


def rebuild_prompt_cache(reference_feat, recent_texts: list[str], recent_feats: list[object]) -> Optional[dict]:
    if not recent_feats:
        if reference_feat is None:
            return None
        return {"ref_audio_feat": reference_feat, "mode": "reference"}
    import torch

    cache = {
        "prompt_text": "".join(recent_texts),
        "audio_feat": torch.cat(recent_feats, dim=0),
        "mode": "continuation",
    }
    if reference_feat is not None:
        cache["ref_audio_feat"] = reference_feat
        cache["mode"] = "ref_continuation"
    return cache


def synthesize_longform(
    *,
    model,
    segments: list[DirectorSegment],
    output_root: Path,
    reference_wav_path: Optional[str] = None,
    prompt_text: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
    rolling_context_segments: int = 3,
    use_continuity: bool = True,
    apply_speed_control: bool = True,
    apply_prompts_to_tts: bool = False,
    max_segment_retries: int = 2,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> LongformResult:
    if not segments:
        raise ValueError("No director segments to synthesize.")

    output_root.mkdir(parents=True, exist_ok=True)
    job_dir = output_root / time.strftime("voxdirector-%Y%m%d-%H%M%S")
    segment_dir = job_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = int(model.tts_model.sample_rate)
    clips: list[np.ndarray] = []
    manifest: list[dict] = []

    tts_model = getattr(model, "tts_model", None)
    can_cache = (
        use_continuity
        and tts_model is not None
        and hasattr(tts_model, "build_prompt_cache")
        and hasattr(tts_model, "generate_with_prompt_cache")
    )

    reference_feat = None
    if can_cache and reference_wav_path:
        initial_cache = tts_model.build_prompt_cache(
            prompt_text=prompt_text or None,
            prompt_wav_path=reference_wav_path if prompt_text else None,
            reference_wav_path=reference_wav_path,
        )
        reference_feat = initial_cache.get("ref_audio_feat")

    recent_texts: list[str] = []
    recent_feats: list[object] = []
    prompt_cache = rebuild_prompt_cache(reference_feat, recent_texts, recent_feats) if can_cache else None

    total_segments = len(segments)
    for current_index, segment in enumerate(segments, 1):
        if progress_callback is not None:
            progress_callback(current_index, total_segments, "generating", segment.text)
        final_text = final_text_for_segment(segment, apply_prompt=apply_prompts_to_tts)
        audio = None
        new_feat_for_cache = None
        last_error: Optional[Exception] = None
        for attempt in range(max(1, max_segment_retries + 1)):
            try:
                attempt_uses_cache = can_cache and attempt == 0
                if attempt_uses_cache:
                    audio_tensor, _tokens, new_feat = tts_model.generate_with_prompt_cache(
                        target_text=final_text,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        max_len=max(48, min(220, len(segment.text) * 4 + 20)),
                        retry_badcase=True,
                        retry_badcase_max_times=2,
                        retry_badcase_ratio_threshold=4.0,
                    )
                    candidate_audio = _to_numpy_audio(audio_tensor)
                    candidate_audio = validate_or_repair_segment_audio(
                        candidate_audio,
                        sample_rate=sample_rate,
                        segment=segment,
                    )
                    new_feat_for_cache = new_feat
                else:
                    candidate_audio = model.generate(
                        text=final_text,
                        reference_wav_path=reference_wav_path,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        normalize=normalize,
                        denoise=denoise and reference_wav_path is not None,
                        max_len=max(48, min(220, len(segment.text) * 4 + 20)),
                        retry_badcase=True,
                        retry_badcase_max_times=2,
                        retry_badcase_ratio_threshold=4.0,
                    )
                    candidate_audio = _to_numpy_audio(candidate_audio)
                    candidate_audio = validate_or_repair_segment_audio(
                        candidate_audio,
                        sample_rate=sample_rate,
                        segment=segment,
                    )
                    new_feat_for_cache = None
                audio = candidate_audio
                break
            except Exception as exc:
                last_error = exc
                # A bad cached continuation can poison following segments. Drop it
                # and retry this segment as reference-only.
                prompt_cache = rebuild_prompt_cache(reference_feat, [], []) if can_cache else None
                recent_texts = []
                recent_feats = []
        if audio is None:
            raise SegmentGenerationError(
                f"Segment {segment.index} failed after retries: {last_error}"
            )

        if can_cache and new_feat_for_cache is not None:
            recent_texts.append(segment.text)
            recent_feats.append(new_feat_for_cache)
            if len(recent_texts) > rolling_context_segments:
                recent_texts = recent_texts[-rolling_context_segments:]
                recent_feats = recent_feats[-rolling_context_segments:]
            prompt_cache = rebuild_prompt_cache(reference_feat, recent_texts, recent_feats)

        if apply_speed_control:
            audio = adjust_speed_if_needed(audio, sample_rate, segment.text, segment.speed)
            audio = validate_or_repair_segment_audio(audio, sample_rate=sample_rate, segment=segment)
        audio = normalize_peak(audio)

        segment_path = segment_dir / f"segment_{segment.index:04d}.wav"
        sf.write(segment_path, audio, sample_rate)
        clips.append(audio)
        duration = len(audio) / float(sample_rate)
        manifest.append(
            {
                **asdict(segment),
                "status": "done",
                "segment_path": str(segment_path),
                "duration_seconds": round(duration, 3),
                "chars_per_second": round(chars_per_second(segment.text, audio, sample_rate), 3),
            }
        )
        if progress_callback is not None:
            progress_callback(current_index, total_segments, "saved", segment.text)

    if progress_callback is not None:
        progress_callback(total_segments, total_segments, "assembling", "")
    full_audio = concat_with_pauses(
        clips,
        [segment.pause_ms for segment in segments],
        sample_rate=sample_rate,
    )
    output_path = job_dir / "voxdirector_longform.wav"
    manifest_path = job_dir / "manifest.json"
    sf.write(output_path, full_audio, sample_rate)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback(total_segments, total_segments, "done", "")

    return LongformResult(
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        segment_dir=str(segment_dir),
        segment_count=len(segments),
        duration_seconds=round(len(full_audio) / float(sample_rate), 3),
    )
