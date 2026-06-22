# VoxDirector Upgrade Notes

VoxDirector is the next-stage direction for VoxCPM Easy Launcher: a long-form narration tool that keeps VoxCPM local inference, but adds script directing, editable segmentation, rolling voice continuity, and final audio assembly.

## What changed

- Added a new `VoxDirector Long-form` tab in `VoxCPM-main/app.py`.
- Added `src/voxcpm/longform.py` for:
  - semantic text splitting;
  - automatic local emotion analysis;
  - per-segment control prompt generation;
  - editable director-table conversion;
  - rolling prompt-cache generation when VoxCPM2 exposes cache APIs;
  - segment WAV writing, pauses, crossfades, gentle speed correction, and final WAV assembly.

## Workflow

1. Paste or upload a long script.
2. Build a Director Table.
3. Manually edit segment text, emotion, speed, prompt, and pause length.
4. Generate long-form audio.
5. Review the final WAV and the JSON manifest.

## Segmentation policy

The splitter avoids cutting through ordinary sentences. It prefers:

1. paragraph boundaries;
2. sentence endings such as `。`, `？`, `！`, `……`, `;`;
3. comma-like pauses only when a sentence is too long;
4. length fallback only for extreme unpunctuated text.

There is no total-script length cap. The per-segment length is a quality and runtime control, not a document limit.

## Continuity policy

When available, VoxDirector uses VoxCPM2 prompt-cache APIs:

- a fixed reference voice anchors the speaker identity;
- the most recent generated segments form a rolling continuation context;
- the rolling window prevents unbounded memory growth on very long scripts.

If prompt-cache APIs are unavailable, generation falls back to ordinary per-segment synthesis.

## Current limitations

- Emotion analysis is rule-based and offline. It is stable and editable, but not as nuanced as an optional future LLM-based director.
- Speed control is engineering-level correction. VoxCPM does not expose a hard speech-rate parameter, so VoxDirector combines prompts, chunk sizing, pause insertion, and gentle time-stretching.
- The first implementation generates the whole job synchronously through Gradio. Resume, cancel, single-segment regeneration, and project save/load should be added next.

## Recommended next steps

- Add project save/load for Director Tables.
- Add per-segment preview and regenerate.
- Add resumable jobs and failure recovery.
- Add optional AI director analysis while keeping manual control.
- Update macOS and Windows installers to package the new source tree.
