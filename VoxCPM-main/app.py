import os
import re
import sys
import json
import html as html_lib
import logging
import numpy as np
import soundfile as sf
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path
import time
import zipfile
from urllib.parse import quote

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.longform import (
    build_director_segments,
    concat_with_pauses,
    rows_to_segments,
    segments_to_rows,
    synthesize_longform,
)
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DIRECTOR_STATUS_COL = 5
DIRECTOR_AUDIO_COL = 6
LEGACY_DIRECTOR_STATUS_COL = 6
LEGACY_DIRECTOR_AUDIO_COL = 7


def director_rows_to_list(rows) -> list[list[object]]:
    if rows is None:
        return []
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    normalized = []
    for row in rows:
        if row is None:
            continue
        row = list(row)
        while len(row) < 7:
            row.append("")
        normalized.append(row)
    return normalized


def director_status_index(row: list[object]) -> int:
    return LEGACY_DIRECTOR_STATUS_COL if len(row) >= 8 else DIRECTOR_STATUS_COL


def director_audio_index(row: list[object]) -> int:
    return LEGACY_DIRECTOR_AUDIO_COL if len(row) >= 8 else DIRECTOR_AUDIO_COL


def gradio_file_url(path: str) -> str:
    return f"/gradio_api/file={quote(str(path), safe='/:')}"


def director_audio_path(row: list[object]) -> str:
    value = str(row[director_audio_index(row)] or "").strip()
    match = re.search(r'data-path="([^"]*)"', value)
    if match:
        return html_lib.unescape(match.group(1)).strip()
    if "<" in value and ">" in value:
        return ""
    return value


def director_audio_cell(path: str = "") -> str:
    path = str(path or "").strip()
    if path:
        escaped_path = html_lib.escape(path, quote=True)
        url = html_lib.escape(gradio_file_url(path), quote=True)
        return (
            '<div class="director-audio-cell" data-path="'
            + escaped_path
            + '">'
            + f'<audio controls preload="none" src="{url}"></audio>'
            + '<div class="director-audio-actions">'
            + f'<a href="{url}" download>下载</a>'
            + '<button type="button" class="vd-regenerate">重新生成</button>'
            + "</div></div>"
        )
    return (
        '<div class="director-audio-cell" data-path="">'
        '<span class="director-audio-empty">未生成</span>'
        '<button type="button" class="vd-regenerate">生成</button>'
        "</div>"
    )


def director_audio_progress_cell(label: str = "生成中") -> str:
    label = html_lib.escape(label, quote=True)
    return (
        '<div class="director-audio-cell director-audio-generating" data-path="">'
        f'<span>{label}</span>'
        '<div class="director-row-progress"><div></div></div>'
        "</div>"
    )


def decorate_director_audio_cells(rows: list[list[object]]) -> list[list[object]]:
    for row in rows:
        while len(row) < 7:
            row.append("")
        row[director_audio_index(row)] = director_audio_cell(director_audio_path(row))
    return rows


def reindex_director_rows(rows: list[list[object]]) -> list[list[object]]:
    for index, row in enumerate(rows, 1):
        while len(row) < 7:
            row.append("")
        row[0] = index
        row[director_audio_index(row)] = director_audio_cell(director_audio_path(row))
    return rows


def invalidate_director_row(row: list[object]) -> list[object]:
    while len(row) < 7:
        row.append("")
    row[director_status_index(row)] = "pending"
    row[director_audio_index(row)] = director_audio_cell("")
    return row


def split_director_row(rows, selected_row: int, editor_text: str) -> tuple[list[list[object]], int, str]:
    rows = reindex_director_rows(director_rows_to_list(rows))
    if not rows:
        return rows, 0, ""
    row_index = max(0, min(int(selected_row or 0), len(rows) - 1))
    raw = (editor_text or rows[row_index][1] or "").strip()
    split_at = -1
    for marker in ("\n", "|", "｜"):
        split_at = raw.find(marker)
        if split_at > 0:
            break
    if split_at <= 0 or split_at >= len(raw) - 1:
        return rows, row_index, raw
    left = raw[:split_at].strip()
    right = raw[split_at + 1 :].strip()
    if not left or not right:
        return rows, row_index, raw
    first = invalidate_director_row(list(rows[row_index]))
    second = invalidate_director_row(list(rows[row_index]))
    first[1] = left
    second[1] = right
    rows[row_index : row_index + 1] = [first, second]
    return reindex_director_rows(rows), row_index, left


def merge_director_rows(rows, selected_row: int, direction: str) -> tuple[list[list[object]], int, str]:
    rows = reindex_director_rows(director_rows_to_list(rows))
    if not rows:
        return rows, 0, ""
    row_index = max(0, min(int(selected_row or 0), len(rows) - 1))
    if direction == "previous":
        if row_index == 0:
            return rows, row_index, str(rows[row_index][1] or "")
        target_index = row_index - 1
        merged_text = f"{rows[target_index][1] or ''}{rows[row_index][1] or ''}".strip()
        rows[target_index][1] = merged_text
        invalidate_director_row(rows[target_index])
        rows.pop(row_index)
        return reindex_director_rows(rows), target_index, merged_text
    if row_index >= len(rows) - 1:
        return rows, row_index, str(rows[row_index][1] or "")
    merged_text = f"{rows[row_index][1] or ''}{rows[row_index + 1][1] or ''}".strip()
    rows[row_index][1] = merged_text
    invalidate_director_row(rows[row_index])
    rows.pop(row_index + 1)
    return reindex_director_rows(rows), row_index, merged_text


def apply_director_keyboard_command(rows, selected_row: int, command_json: str):
    rows = reindex_director_rows(director_rows_to_list(rows))
    if not rows:
        return rows, 0, None, None, ""
    try:
        command = json.loads(command_json or "{}")
    except json.JSONDecodeError:
        command = {}
    row_index = max(0, min(int(selected_row or 0), len(rows) - 1))
    action = command.get("action")
    if action == "split":
        before = str(command.get("before") or "").strip()
        after = str(command.get("after") or "").strip()
        if not before or not after:
            return rows, row_index, None, None, ""
        first = invalidate_director_row(list(rows[row_index]))
        second = invalidate_director_row(list(rows[row_index]))
        first[1] = before
        second[1] = after
        rows[row_index : row_index + 1] = [first, second]
        rows = reindex_director_rows(rows)
        return rows, row_index, None, None, f"已拆分为第 {row_index + 1} / {row_index + 2} 段。"
    if action == "merge_previous":
        rows, row_index, _text = merge_director_rows(rows, row_index, "previous")
        return rows, row_index, None, None, f"已合并到第 {row_index + 1} 段。"
    return rows, row_index, None, None, ""

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_OPERATION_TIPS_ZH = (
    "### 推荐工作流\n\n"
    "1. 粘贴一整篇稿件，或上传 UTF-8 文本文件。单句也直接粘贴到这里使用。\n"
    "2. 上传 10-30 秒干净参考音频，尽量无人声重叠、无背景音乐、无明显混响。\n"
    "3. 点击“自动识别参考音频文字”，检查识别结果；参考文本越准，克隆语速和节奏越稳。\n"
    "4. 生成导演表后，直接在表格里改文字、提示词、停顿和语速。\n"
    "5. 在文本单元格中按 Enter 可从光标处分成两行；在一行开头按 Backspace 可合并到上一行。\n"
    "6. 先生成全部分段，再在“分段音频”栏逐条试听、下载或重新生成；满意后再合并完整音频。\n\n"
    "### 分段建议\n\n"
    "- VoxCPM2 更适合短片段，中文解说通常建议每段约 30-80 字；长句尽量按句号、问号、叹号、省略号或自然停顿拆开。\n"
    "- 不要为了卡字数硬切半句话。语义完整的短句，通常比机械等长切片更自然。\n"
    "- 每段之间默认至少保留 1 秒停顿；你也可以在表格里调大“停顿 ms”。\n\n"
    "### 提示词与能力边界\n\n"
    "- “提示词”主要帮助描述情绪、风格和语境，但它不是强控制器；语速、情绪不会像剪辑软件参数那样绝对执行。\n"
    "- 语速优先由参考音频和分段长度决定。想要慢速解说，参考音频本身就要慢、稳、清楚。\n"
    "- 默认不会把每段提示词送进 TTS，避免模型把提示词读出来；高级设置里的实验开关请谨慎使用。\n"
    "- 滚动连续上下文可能让前后更连贯，但也可能累积音质劣化；当前默认关闭，更适合逐段试听后抽卡。\n"
    "- 本地模型生成长稿会很慢，质量也会有波动。导演表的价值就是让你逐段修、逐段重抽，而不是一次赌完整长音频。\n\n"
    "### 方言使用\n\n"
    "- 要生成方言，目标文本最好直接写成方言表达，而不是只在提示词里写“用粤语/河南话”。\n"
    "- 例如粤语可写“伙計，唔該一個A餐，凍奶茶少甜！”，河南话可写“恁这是弄啥嘞？晌午吃啥饭？”\n"
    "- 可以先用 AI 助手把普通话改写成方言文本，再人工检查语气和用词。参考音频如果也是同一种方言，稳定性会更好。"
)

_OPERATION_TIPS_EN = (
    "### Recommended Workflow\n\n"
    "1. Paste a full script, upload a UTF-8 text file, or paste a single sentence here.\n"
    "2. Upload a clean 10-30 second reference voice with no overlapping speech, music, or heavy room echo.\n"
    "3. Auto-transcribe the reference audio and check the transcript; accurate reference text helps match pace and rhythm.\n"
    "4. Build the director table, then edit text, prompts, pauses, and speed directly in the table.\n"
    "5. Press Enter inside a text cell to split at the cursor; press Backspace at the start of a row to merge into the previous row.\n"
    "6. Generate all segments, audition each row in the Segment Audio column, regenerate weak rows, then merge the final audio.\n\n"
    "### Segmentation\n\n"
    "- VoxCPM2 works best on short clips. For Chinese narration, roughly 30-80 characters per segment is usually a good starting point.\n"
    "- Prefer sentence boundaries and natural pauses over mechanically equal chunks.\n"
    "- Each segment keeps at least 1 second of pause by default; adjust Pause ms in the table when a stronger break is needed.\n\n"
    "### Prompting And Limits\n\n"
    "- Segment prompts describe emotion, style, and context, but they are not hard controls like parameters in an editor.\n"
    "- Speaking speed is mostly anchored by the reference voice and segment wording. For slow narration, use a slow, steady reference.\n"
    "- Segment prompts are not sent into TTS by default to avoid the model reading them aloud; use the experimental switch carefully.\n"
    "- Rolling continuity can improve flow, but may also accumulate quality drift, so it is off by default.\n"
    "- Local long-form generation is slow and quality can vary. The director table is designed for auditioning and regenerating individual rows.\n\n"
    "### Dialects\n\n"
    "- For dialects, write the target text in the dialect itself instead of only asking for a dialect in the prompt.\n"
    "- You can ask an AI assistant to rewrite Mandarin into the dialect, then manually check wording and tone.\n"
    "- A reference voice in the same dialect usually improves consistency."
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

UI_TEXT = {
    "zh": {
        "app_title": "DubCue 配音导演",
        "language_button": "语言切换：中文",
        "tips": "操作技巧",
        "tips_markdown": _OPERATION_TIPS_ZH,
        "usage": _USAGE_INSTRUCTIONS_ZH,
        "examples": _EXAMPLES_FOOTER_ZH,
        "reference_audio": "🎤 参考音频（可选，上传后用于克隆）",
        "ultimate_mode": "🎙️ 极致克隆模式（基于参考音频文本）",
        "ultimate_info": "自动识别参考音频文本，开启后将禁用 Control Instruction",
        "prompt_text": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control": "🎛️ Control Instruction（可选）",
        "control_placeholder": "如：年轻女性，温柔甜美 / 暴躁老哥，语速飞快",
        "target_text": "✍️ Target Text — 要合成的目标文本",
        "advanced": "⚙️ 高级设置",
        "denoise": "参考音频降噪增强",
        "denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg": "CFG（引导强度）",
        "cfg_info": "数值越高越贴合提示/参考音色；数值越低风格更自由",
        "dit": "LocDiT 流匹配迭代步数",
        "dit_info": "步数越多可能质量更好，但速度更慢",
        "generate": "🔊 开始生成",
        "generated_audio": "生成结果",
        "long_script": "长文本稿件",
        "long_script_placeholder": "粘贴纪录片旁白、有声书、解说词或任意长文本。",
        "text_file": "或上传 UTF-8 文本文件",
        "reference_voice": "参考声音",
        "reference_transcript": "参考音频文本（可选，用于匹配参考语速）",
        "recognize_reference": "自动识别参考音频文字",
        "max_chars": "每段最大字数",
        "style": "整体配音风格",
        "style_notes": "全局风格补充",
        "style_notes_placeholder": "如：冷静的自然纪录片，语速缓慢，克制",
        "variation": "情绪变化幅度",
        "allow_split": "允许对超长句按自然停顿二次切分",
        "build_table": "生成导演表",
        "status": "状态",
        "table": "Director Table / 导演表",
        "table_headers": ["序号", "文本", "语速", "提示词", "停顿 ms", "状态", "分段音频"],
        "long_settings": "长文本生成设置",
        "rolling": "滚动连续上下文段数（实验性）",
        "continuity": "实验性滚动 prompt cache（可能累积劣化，默认关闭）",
        "speed_control": "自然语速守卫（太快则重试，不做后期变速）",
        "apply_prompts": "实验性：把每段提示词送入 TTS（可能被读出来，默认关闭）",
        "generate_long": "生成全部分段并回写导演表",
        "selected_segment": "当前选中分段",
        "segment_preview": "试听当前分段",
        "segment_file": "下载当前分段",
        "regenerate_segment": "重新生成当前分段",
        "merge_segments": "合并已生成分段",
        "download_segments": "下载全部分段",
        "segments_zip": "分段音频包",
        "final_audio": "完整长文本音频",
        "manifest": "生成清单",
        "generation_status": "生成状态",
        "ready": "导演表已生成：{count} 段。",
        "need_script": "请先粘贴或上传长文本稿件。",
        "no_segments": "没有找到有效文本分段。",
        "done": "完成：{count} 段，{duration:.1f} 秒。Manifest: {manifest}",
        "preparing": "准备生成 {count} 段...",
        "progress_generating": "正在生成第 {current}/{total} 段：{preview}",
        "progress_saved": "已保存第 {current}/{total} 段",
        "progress_assembling": "正在拼接完整音频",
        "progress_done": "生成完成",
    },
    "en": {
        "app_title": "DubCue Dubbing Director",
        "language_button": "Language: English",
        "tips": "Tips",
        "tips_markdown": _OPERATION_TIPS_EN,
        "usage": _USAGE_INSTRUCTIONS_EN,
        "examples": _EXAMPLES_FOOTER_EN,
        "reference_audio": "🎤 Reference Audio (optional, upload for cloning)",
        "ultimate_mode": "🎙️ Ultimate Cloning Mode (reference transcript)",
        "ultimate_info": "Auto-transcribes the reference audio. Control Instruction is disabled when active.",
        "prompt_text": "Reference transcript (auto-filled via ASR, editable)",
        "prompt_placeholder": "The transcript of your reference audio will appear here …",
        "control": "🎛️ Control Instruction (optional)",
        "control_placeholder": "e.g. warm young woman, slow and gentle",
        "target_text": "✍️ Target Text — content to speak",
        "advanced": "⚙️ Advanced Settings",
        "denoise": "Reference audio enhancement",
        "denoise_info": "Apply ZipEnhancer denoising before cloning",
        "normalize": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg": "CFG (guidance scale)",
        "cfg_info": "Higher means closer to prompt/reference; lower means freer variation",
        "dit": "LocDiT flow-matching steps",
        "dit_info": "More steps may improve quality, but generation is slower",
        "generate": "🔊 Generate Speech",
        "generated_audio": "Generated Audio",
        "long_script": "Long Script",
        "long_script_placeholder": "Paste documentary narration, audiobook text, or any long script.",
        "text_file": "Or upload a UTF-8 text file",
        "reference_voice": "Reference Voice",
        "reference_transcript": "Reference transcript (optional, used to match reference speed)",
        "recognize_reference": "Auto-transcribe reference audio",
        "max_chars": "Maximum characters per segment",
        "style": "Overall narration style",
        "style_notes": "Global style notes",
        "style_notes_placeholder": "e.g. calm nature documentary, slow and restrained",
        "variation": "Emotion variation",
        "allow_split": "Allow splitting very long sentences at natural pauses",
        "build_table": "Build Director Table",
        "status": "Status",
        "table": "Director Table",
        "table_headers": ["Index", "Text", "Speed", "Prompt", "Pause ms", "Status", "Segment Audio"],
        "long_settings": "Long-form generation settings",
        "rolling": "Rolling continuity segments (experimental)",
        "continuity": "Experimental rolling prompt cache (may accumulate drift, off by default)",
        "speed_control": "Natural speed guard (retry if too fast, no time-stretch)",
        "apply_prompts": "Experimental: send segment prompts into TTS (may be spoken, off by default)",
        "generate_long": "Generate All Segments into Director Table",
        "selected_segment": "Selected segment",
        "segment_preview": "Preview Selected Segment",
        "segment_file": "Download Selected Segment",
        "regenerate_segment": "Regenerate Selected Segment",
        "merge_segments": "Merge Generated Segments",
        "download_segments": "Download All Segments",
        "segments_zip": "Segment Audio Package",
        "final_audio": "Final long-form audio",
        "manifest": "Manifest",
        "generation_status": "Generation status",
        "ready": "Director table ready: {count} segments.",
        "need_script": "Please paste or upload a long-form script first.",
        "no_segments": "No valid text segments were found.",
        "done": "Done: {count} segments, {duration:.1f}s. Manifest: {manifest}",
        "preparing": "Preparing {count} segments...",
        "progress_generating": "Generating segment {current}/{total}: {preview}",
        "progress_saved": "Saved segment {current}/{total}",
        "progress_assembling": "Assembling final audio",
        "progress_done": "Done",
    },
}


def ui_text(lang: str, key: str):
    lang = lang if lang in UI_TEXT else "zh"
    return UI_TEXT[lang][key]

_CUSTOM_CSS = """
.top-bar {
    position: relative;
    min-height: 72px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    margin: 0.25rem 0 0.75rem 0;
    padding-right: 132px;
}
.top-bar > .top-bar {
    position: static;
    width: 100%;
    min-width: 100%;
    min-height: 72px;
    margin: 0;
    padding: 0;
    flex: 1 1 100%;
}
.logo-container {
    text-align: left;
    margin: 0;
}
.logo-container img {
    height: 64px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}
.language-switch {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    left: auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: 132px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 4px 9px !important;
    margin: 0 !important;
    white-space: nowrap !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    line-height: 1 !important;
    z-index: 2;
}
@media (max-width: 720px) {
    .top-bar {
        min-height: 74px;
        align-items: center;
        padding-top: 0;
        padding-right: 112px;
    }
    .language-switch {
        max-width: 108px !important;
        height: 28px !important;
        min-height: 28px !important;
        padding: 3px 7px !important;
        font-size: 11px !important;
    }
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
.director-hidden-command {
    display: none !important;
}
.director-audio-cell {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 260px;
}
.director-audio-cell audio {
    width: 160px;
    height: 32px;
}
.director-audio-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.director-audio-actions a,
.director-audio-cell button {
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 4px 8px;
    background: var(--button-secondary-background-fill);
    color: var(--body-text-color);
    cursor: pointer;
    font-size: 12px;
    text-decoration: none;
    white-space: nowrap;
}
.director-audio-empty {
    color: var(--body-text-color-subdued);
    white-space: nowrap;
}
.director-audio-generating {
    flex-direction: column;
    align-items: stretch;
    gap: 5px;
}
.director-row-progress {
    width: 100%;
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--block-border-color);
}
.director-row-progress div {
    width: 45%;
    height: 100%;
    border-radius: inherit;
    background: var(--color-accent);
    animation: director-progress-slide 1.1s ease-in-out infinite;
}
@keyframes director-progress-slide {
    0% { transform: translateX(-110%); }
    100% { transform: translateX(250%); }
}
"""

_DIRECTOR_KEYBOARD_HEAD = """
<script>
(() => {
  if (window.__dubcueKeyboardInstalled) return;
  window.__dubcueKeyboardInstalled = true;

  function editableValue(target) {
    if (target.isContentEditable) return target.innerText || "";
    return typeof target.value === "string" ? target.value : "";
  }

  function editableSelection(target, value) {
    if (!target.isContentEditable) {
      const start = Number.isInteger(target.selectionStart) ? target.selectionStart : value.length;
      const end = Number.isInteger(target.selectionEnd) ? target.selectionEnd : start;
      return { start, end };
    }
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return { start: value.length, end: value.length };
    const range = selection.getRangeAt(0);
    const pre = range.cloneRange();
    pre.selectNodeContents(target);
    pre.setEnd(range.startContainer, range.startOffset);
    const start = pre.toString().length;
    return { start, end: start };
  }

  function cellColumn(target) {
    const cell = target.closest("td,[role='gridcell'],[data-testid*='cell'],.cell");
    if (!cell) return null;
    const aria = cell.getAttribute("aria-colindex");
    if (aria && !Number.isNaN(Number(aria))) return Number(aria) - 1;
    const dataCol = cell.getAttribute("data-col") || cell.getAttribute("data-column-index");
    if (dataCol && !Number.isNaN(Number(dataCol))) return Number(dataCol);
    const row = cell.parentElement;
    if (!row) return null;
    const cells = Array.from(row.children).filter((child) => {
      const tag = child.tagName ? child.tagName.toLowerCase() : "";
      return tag === "td" || child.getAttribute("role") === "gridcell" || child.className.toString().includes("cell");
    });
    const index = cells.indexOf(cell);
    return index >= 0 ? index : null;
  }

  function findTextEditor(event) {
    const target = event.target;
    if (!target || !target.closest) return null;
    const table = target.closest("#director-table");
    if (!table) return null;
    const editable = target.matches("textarea,input,[contenteditable='true']") || target.isContentEditable;
    if (!editable) return null;
    const col = cellColumn(target);
    if (col !== 1) return null;
    const value = editableValue(target);
    const selection = editableSelection(target, value);
    return { value, start: selection.start, end: selection.end };
  }

  function sendCommand(command) {
    const box = document.querySelector("#director-keyboard-command textarea, #director-keyboard-command input");
    const trigger = document.querySelector("#director-keyboard-trigger button, #director-keyboard-trigger");
    if (!box || !trigger) return false;
    box.value = JSON.stringify(command);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.dispatchEvent(new Event("change", { bubbles: true }));
    setTimeout(() => trigger.click(), 0);
    return true;
  }

  document.addEventListener("keydown", (event) => {
    const editor = findTextEditor(event);
    if (!editor) return;
    if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (editor.start <= 0 || editor.start >= editor.value.length) return;
      event.preventDefault();
      event.stopPropagation();
      sendCommand({
        action: "split",
        before: editor.value.slice(0, editor.start),
        after: editor.value.slice(editor.end),
      });
      return;
    }
    if (event.key === "Backspace" && editor.start === 0 && editor.end === 0) {
      event.preventDefault();
      event.stopPropagation();
      sendCommand({ action: "merge_previous" });
    }
  }, true);

  document.addEventListener("click", (event) => {
    const button = event.target && event.target.closest ? event.target.closest(".vd-regenerate") : null;
    if (!button) return;
    const table = button.closest("#director-table");
    if (!table) return;
    event.preventDefault();
    event.stopPropagation();
    const cell = button.closest("td,[role='gridcell'],[data-testid*='cell'],.cell");
    const row = cell ? cell.parentElement : null;
    if (row) {
      const cells = Array.from(row.children).filter((child) => {
        const tag = child.tagName ? child.tagName.toLowerCase() : "";
        return tag === "td" || child.getAttribute("role") === "gridcell" || child.className.toString().includes("cell");
      });
      const first = cells[0];
      if (first) first.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
    setTimeout(() => {
      const trigger = document.querySelector("#director-regenerate-trigger button, #director-regenerate-trigger");
      if (trigger) trigger.click();
    }, 50);
  }, true);
})();
</script>
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------

class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
            load_denoiser=False,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(
            f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}"
        )
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
    ) -> Tuple[int, np.ndarray]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)

    def build_director_table(
        self,
        language: str,
        text_input: str,
        text_file,
        max_chars: int,
        base_style: str,
        user_style: str,
        variation: str,
        allow_sentence_split: bool,
    ):
        text = (text_input or "").strip()
        if text_file is not None:
            file_path = getattr(text_file, "name", text_file)
            try:
                text = Path(file_path).read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                text = Path(file_path).read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(ui_text(language, "need_script"))
        segments = build_director_segments(
            text,
            max_chars=int(max_chars),
            base_style=base_style,
            user_style=user_style,
            variation=variation,
            allow_sentence_split=allow_sentence_split,
        )
        if not segments:
            raise ValueError(ui_text(language, "no_segments"))
        return decorate_director_audio_cells(segments_to_rows(segments)), ui_text(language, "ready").format(count=len(segments))

    def generate_longform_audio(
        self,
        language: str,
        director_rows,
        reference_wav_path_input: Optional[str],
        prompt_text: str,
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int,
        rolling_context_segments: int,
        use_continuity: bool,
        apply_speed_control: bool,
        apply_prompts_to_tts: bool,
        progress=gr.Progress(),
    ):
        rows = decorate_director_audio_cells(director_rows_to_list(director_rows))
        segments = rows_to_segments(rows)
        if not segments:
            raise ValueError("The director table is empty.")

        current_model = self.get_or_load_voxcpm()
        output_root = Path.cwd() / "outputs" / "dubcue"
        manifest_items: list[dict] = []
        total = len(segments)
        manifest_path = ""

        for row_index, segment in enumerate(segments):
            segment.index = row_index + 1
            rows[row_index][director_status_index(rows[row_index])] = "generating"
            rows[row_index][director_audio_index(rows[row_index])] = director_audio_progress_cell(
                f"生成中 {row_index + 1}/{total}"
            )
            progress(row_index / max(total, 1), desc=f"正在生成第 {row_index + 1}/{total} 段")
            yield None, manifest_path or None, "", rows

            try:
                result = synthesize_longform(
                    model=current_model,
                    segments=[segment],
                    output_root=output_root,
                    reference_wav_path=reference_wav_path_input if reference_wav_path_input else None,
                    prompt_text=(prompt_text or "").strip(),
                    cfg_value=cfg_value_input,
                    inference_timesteps=int(inference_timesteps),
                    normalize=do_normalize,
                    denoise=denoise,
                    rolling_context_segments=0,
                    use_continuity=use_continuity and int(rolling_context_segments) > 0,
                    apply_speed_control=apply_speed_control,
                    apply_prompts_to_tts=apply_prompts_to_tts,
                )
                item = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))[0]
                item["index"] = row_index + 1
                manifest_items.append(item)
                rows[row_index][director_status_index(rows[row_index])] = (
                    "done ⚠" if item.get("speed_warning") else "done"
                )
                rows[row_index][director_audio_index(rows[row_index])] = director_audio_cell(
                    item.get("segment_path", "")
                )
            except Exception as exc:
                logger.exception("Director segment generation failed")
                rows[row_index][director_status_index(rows[row_index])] = "failed"
                rows[row_index][director_audio_index(rows[row_index])] = director_audio_cell("")
                manifest_items.append(
                    {
                        "index": row_index + 1,
                        "text": segment.text,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            yield None, manifest_path or None, "", rows

        progress(1, desc=ui_text(language, "progress_assembling"))
        output_root.mkdir(parents=True, exist_ok=True)
        final_dir = output_root / time.strftime("director-session-%Y%m%d-%H%M%S")
        final_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = str(final_dir / "manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest_items, ensure_ascii=False, indent=2), encoding="utf-8")

        final_audio = None
        message = ""
        if all(str(row[director_status_index(row)]).startswith("done") for row in rows):
            final_audio, message = self.merge_director_segments(rows)
        else:
            failed = [str(item["index"]) for item in manifest_items if item.get("status") == "failed"]
            message = f"部分分段生成失败：{', '.join(failed)}" if failed else ""
        yield final_audio, manifest_path, message, rows

    def generate_director_segment(
        self,
        language: str,
        director_rows,
        selected_row: int,
        reference_wav_path_input: Optional[str],
        prompt_text: str,
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int,
        use_continuity: bool,
        apply_speed_control: bool,
        apply_prompts_to_tts: bool,
        progress=gr.Progress(),
    ):
        rows = director_rows_to_list(director_rows)
        if not rows:
            raise ValueError("The director table is empty.")
        row_index = max(0, min(int(selected_row or 0), len(rows) - 1))
        segments = rows_to_segments([rows[row_index]])
        if not segments:
            raise ValueError("The selected row has no valid text.")
        segment = segments[0]
        try:
            segment.index = int(float(rows[row_index][0] or row_index + 1))
        except (TypeError, ValueError):
            segment.index = row_index + 1

        current_model = self.get_or_load_voxcpm()
        output_root = Path.cwd() / "outputs" / "dubcue"
        progress(0, desc=f"Generating segment {segment.index}...")
        result = synthesize_longform(
            model=current_model,
            segments=[segment],
            output_root=output_root,
            reference_wav_path=reference_wav_path_input if reference_wav_path_input else None,
            prompt_text=(prompt_text or "").strip(),
            cfg_value=cfg_value_input,
            inference_timesteps=int(inference_timesteps),
            normalize=do_normalize,
            denoise=denoise,
            rolling_context_segments=0,
            use_continuity=use_continuity,
            apply_speed_control=apply_speed_control,
            apply_prompts_to_tts=apply_prompts_to_tts,
        )
        progress(1, desc="Segment saved")
        manifest_items = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        item = manifest_items[0] if manifest_items else {}
        segment_path = item.get("segment_path", result.output_path)
        rows[row_index][director_status_index(rows[row_index])] = (
            "done ⚠" if item.get("speed_warning") else "done"
        )
        rows[row_index][director_audio_index(rows[row_index])] = director_audio_cell(segment_path)
        status = f"第 {segment.index} 段已生成：{segment_path}"
        return rows, segment_path, segment_path, status

    def merge_director_segments(self, director_rows):
        rows = director_rows_to_list(director_rows)
        segments = rows_to_segments(rows)
        if not segments:
            raise ValueError("The director table is empty.")
        clips = []
        pauses = []
        sample_rate = None
        missing = []
        for row, segment in zip(rows, segments):
            path = director_audio_path(row)
            if not path or not Path(path).exists():
                missing.append(str(segment.index))
                continue
            audio, sr = sf.read(path, dtype="float32")
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            sample_rate = int(sr) if sample_rate is None else sample_rate
            if int(sr) != sample_rate:
                raise ValueError("Segment sample rates do not match.")
            clips.append(audio)
            pauses.append(segment.pause_ms)
        if missing:
            raise ValueError(f"这些段落还没有生成音频：{', '.join(missing)}")
        output_dir = Path.cwd() / "outputs" / "dubcue" / time.strftime("director-merge-%Y%m%d-%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        full_audio = concat_with_pauses(clips, pauses, sample_rate or 24000, min_pause_ms=1000)
        output_path = output_dir / "dubcue_merged.wav"
        sf.write(output_path, full_audio, sample_rate or 24000)
        return str(output_path), f"已合并 {len(clips)} 段：{output_path}"

    def zip_director_segments(self, director_rows):
        rows = director_rows_to_list(director_rows)
        paths = []
        for row in rows:
            path = director_audio_path(row)
            if path and Path(path).exists():
                paths.append(Path(path))
        if not paths:
            raise ValueError("还没有可下载的分段音频。")
        output_dir = Path.cwd() / "outputs" / "dubcue" / time.strftime("director-segments-%Y%m%d-%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / "dubcue_segments.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=path.name)
        return str(zip_path)


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    assets_dir = Path(__file__).resolve().parent / "assets"
    gr.set_static_paths(paths=[assets_dir])
    logo_src = "/gradio_api/file=" + quote(str(assets_dir / "voxcpm_logo.png"))

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
        )
        return (sr, wav_np)

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    def _run_long_reference_asr(audio_path):
        """Auto-fill the long-form reference transcript from reference audio."""
        if not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on long-form reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"Long-form ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"Long-form ASR recognition failed: {e}")
            return gr.update(value="")

    def _select_director_row(director_rows, evt: gr.SelectData):
        rows = director_rows_to_list(director_rows)
        row_index = 0
        if isinstance(evt.index, (list, tuple)):
            row_index = int(evt.index[0])
        elif evt.index is not None:
            row_index = int(evt.index)
        if not rows:
            return 0, None, None, ""
        row_index = max(0, min(row_index, len(rows) - 1))
        row = rows[row_index]
        audio_path = director_audio_path(row)
        audio_value = audio_path if audio_path and Path(audio_path).exists() else None
        status = f"第 {row[0]} 段 / Segment {row[0]}：{row[director_status_index(row)] or 'pending'}"
        return row_index, audio_value, audio_value, status

    def _split_segment(director_rows, selected_row):
        rows = director_rows_to_list(director_rows)
        row_index = max(0, min(int(selected_row or 0), len(rows) - 1)) if rows else 0
        current_text = str(rows[row_index][1] or "") if rows else ""
        rows, row_index, _editor_text = split_director_row(director_rows, selected_row, current_text)
        status = f"已拆分。当前选中第 {row_index + 1} 段。" if rows else ""
        return rows, row_index, None, None, status

    def _merge_segment_previous(director_rows, selected_row):
        rows, row_index, _editor_text = merge_director_rows(director_rows, selected_row, "previous")
        status = f"已合并。当前选中第 {row_index + 1} 段。" if rows else ""
        return rows, row_index, None, None, status

    def _merge_segment_next(director_rows, selected_row):
        rows, row_index, _editor_text = merge_director_rows(director_rows, selected_row, "next")
        status = f"已合并。当前选中第 {row_index + 1} 段。" if rows else ""
        return rows, row_index, None, None, status

    def _apply_language(language: str):
        t = UI_TEXT[language if language in UI_TEXT else "zh"]
        return [
            gr.update(value=f"## {t['app_title']}"),
            gr.update(value=t["language_button"]),
            gr.update(label=t["tips"]),
            gr.update(value=t["tips_markdown"]),
            gr.update(value=f"### {t['long_script']}"),
            gr.update(label=t["long_script"], placeholder=t["long_script_placeholder"]),
            gr.update(label=t["text_file"]),
            gr.update(label=t["reference_voice"]),
            gr.update(label=t["reference_transcript"]),
            gr.update(value=t["recognize_reference"]),
            gr.update(label=t["max_chars"]),
            gr.update(label=t["style"]),
            gr.update(label=t["style_notes"], placeholder=t["style_notes_placeholder"]),
            gr.update(label=t["variation"]),
            gr.update(label=t["allow_split"]),
            gr.update(value=t["build_table"]),
            gr.update(label=t["status"]),
            gr.update(label=t["table"], headers=t["table_headers"]),
            gr.update(label=t["long_settings"]),
            gr.update(label=t["rolling"]),
            gr.update(label=t["continuity"]),
            gr.update(label=t["speed_control"]),
            gr.update(label=t["apply_prompts"]),
            gr.update(value=t["generate_long"]),
            gr.update(label=t["selected_segment"]),
            gr.update(label=t["segment_preview"]),
            gr.update(label=t["segment_file"]),
            gr.update(value=t["regenerate_segment"]),
            gr.update(value=t["merge_segments"]),
            gr.update(value=t["download_segments"]),
            gr.update(label=t["segments_zip"]),
            gr.update(label=t["final_audio"]),
            gr.update(label=t["manifest"], visible=False),
            gr.update(label=t["generation_status"], visible=False),
        ]

    def _toggle_language(language: str):
        next_language = "en" if language == "zh" else "zh"
        return [next_language, *_apply_language(next_language)]

    with gr.Blocks(head=_DIRECTOR_KEYBOARD_HEAD) as interface:
        with gr.Group(elem_classes=["top-bar"]):
            gr.HTML(
                '<div class="logo-container">'
                f'<img src="{logo_src}" alt="DubCue Logo">'
                "</div>"
            )
            language = gr.State("zh")
            language_btn = gr.Button(
                ui_text("zh", "language_button"),
                variant="secondary",
                elem_classes=["language-switch"],
            )

        app_title_markdown = gr.Markdown("## " + ui_text("zh", "app_title"))
        with gr.Accordion(ui_text("zh", "tips"), open=False) as tips_accordion:
            tips_markdown = gr.Markdown(ui_text("zh", "tips_markdown"))

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    script_module_title = gr.Markdown("### " + ui_text("zh", "long_script"))
                    long_text = gr.Textbox(
                        label=ui_text("zh", "long_script"),
                        placeholder=ui_text("zh", "long_script_placeholder"),
                        lines=12,
                        show_label=False,
                    )
                    long_text_file = gr.File(
                        label=ui_text("zh", "text_file"),
                        file_types=[".txt", ".md"],
                    )

            with gr.Column(scale=1):
                max_chars = gr.Slider(
                    minimum=30,
                    maximum=120,
                    value=70,
                    step=5,
                    label=ui_text("zh", "max_chars"),
                )
                base_style = gr.Dropdown(
                    choices=[
                        ("Documentary", "documentary"),
                        ("News", "news"),
                        ("Story", "story"),
                        ("Commercial", "commercial"),
                    ],
                    value="documentary",
                    label=ui_text("zh", "style"),
                )
                user_style = gr.Textbox(
                    value="",
                    label=ui_text("zh", "style_notes"),
                    placeholder=ui_text("zh", "style_notes_placeholder"),
                    lines=2,
                )
                emotion_variation = gr.Radio(
                    choices=[
                        ("Low", "low"),
                        ("Medium", "medium"),
                        ("High", "high"),
                    ],
                    value="medium",
                    label=ui_text("zh", "variation"),
                )
                allow_sentence_split = gr.Checkbox(
                    value=True,
                    label=ui_text("zh", "allow_split"),
                    elem_classes=["switch-toggle"],
                )
                build_table_btn = gr.Button(
                    ui_text("zh", "build_table"),
                    variant="primary",
                )
                director_status = gr.Textbox(
                    value="",
                    visible=False,
                )

        director_table = gr.Dataframe(
            headers=[
                *ui_text("zh", "table_headers"),
            ],
            datatype=["number", "str", "str", "str", "number", "str", "html"],
            interactive=True,
            wrap=True,
            label=ui_text("zh", "table"),
            elem_id="director-table",
            column_widths=["64px", "260px", "78px", "260px", "88px", "90px", "320px"],
        )

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    long_reference_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label=ui_text("zh", "reference_voice"),
                    )
                with gr.Column(scale=1):
                    long_prompt_text = gr.Textbox(
                        value="",
                        label=ui_text("zh", "reference_transcript"),
                        lines=2,
                    )
                    recognize_long_reference_btn = gr.Button(
                        ui_text("zh", "recognize_reference"),
                        variant="secondary",
                    )

        selected_director_row = gr.State(0)
        director_keyboard_command = gr.Textbox(
            value="",
            elem_id="director-keyboard-command",
            elem_classes=["director-hidden-command"],
            show_label=False,
            container=False,
        )
        director_keyboard_trigger = gr.Button(
            "",
            elem_id="director-keyboard-trigger",
            elem_classes=["director-hidden-command"],
            visible=True,
        )
        director_regenerate_trigger = gr.Button(
            "",
            elem_id="director-regenerate-trigger",
            elem_classes=["director-hidden-command"],
            visible=True,
        )

        with gr.Row(visible=False):
            with gr.Column(scale=1):
                selected_segment_status = gr.Textbox(
                    label=ui_text("zh", "selected_segment"),
                    value="",
                    interactive=False,
                )
                segment_audio_preview = gr.Audio(
                    label=ui_text("zh", "segment_preview"),
                    type="filepath",
                )
            with gr.Column(scale=1):
                segment_file_output = gr.File(label=ui_text("zh", "segment_file"))
                regenerate_segment_btn = gr.Button(
                    ui_text("zh", "regenerate_segment"),
                    variant="secondary",
                )
                zip_segments_btn = gr.Button(ui_text("zh", "download_segments"))
                segments_zip_output = gr.File(label=ui_text("zh", "segments_zip"))

        with gr.Row():
            with gr.Column():
                with gr.Accordion(ui_text("zh", "long_settings"), open=False) as long_advanced:
                    long_denoise = gr.Checkbox(
                        value=False,
                        label=ui_text("zh", "denoise"),
                        elem_classes=["switch-toggle"],
                        info=ui_text("zh", "denoise_info"),
                    )
                    long_normalize = gr.Checkbox(
                        value=False,
                        label=ui_text("zh", "normalize"),
                        elem_classes=["switch-toggle"],
                        info=ui_text("zh", "normalize_info"),
                    )
                    long_cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=ui_text("zh", "cfg"),
                        info=ui_text("zh", "cfg_info"),
                    )
                    long_dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=ui_text("zh", "dit"),
                        info=ui_text("zh", "dit_info"),
                    )
                    rolling_context = gr.Slider(
                        minimum=0,
                        maximum=8,
                        value=0,
                        step=1,
                        label=ui_text("zh", "rolling"),
                    )
                    use_continuity = gr.Checkbox(
                        value=False,
                        label=ui_text("zh", "continuity"),
                        elem_classes=["switch-toggle"],
                    )
                    apply_speed_control = gr.Checkbox(
                        value=True,
                        label=ui_text("zh", "speed_control"),
                        elem_classes=["switch-toggle"],
                    )
                    apply_prompts_to_tts = gr.Checkbox(
                        value=False,
                        label=ui_text("zh", "apply_prompts"),
                        elem_classes=["switch-toggle"],
                    )
                generate_long_btn = gr.Button(
                    ui_text("zh", "generate_long"),
                    variant="primary",
                    size="lg",
                )
                merge_segments_btn = gr.Button(
                    ui_text("zh", "merge_segments"),
                    variant="secondary",
                )

            with gr.Column():
                long_audio_output = gr.Audio(label=ui_text("zh", "final_audio"))
                manifest_output = gr.File(
                    label=ui_text("zh", "manifest"),
                    visible=False,
                )
                long_generation_status = gr.Textbox(
                    label=ui_text("zh", "generation_status"),
                    value="",
                    interactive=False,
                    visible=False,
                )

        build_table_btn.click(
            fn=demo.build_director_table,
            inputs=[
                language,
                long_text,
                long_text_file,
                max_chars,
                base_style,
                user_style,
                emotion_variation,
                allow_sentence_split,
            ],
            outputs=[director_table, director_status],
            show_progress=True,
            api_name="build_director_table",
        )

        recognize_long_reference_btn.click(
            fn=_run_long_reference_asr,
            inputs=[long_reference_wav],
            outputs=[long_prompt_text],
            show_progress=True,
            api_name="recognize_long_reference",
        )

        director_table.select(
            fn=_select_director_row,
            inputs=[director_table],
            outputs=[
                selected_director_row,
                segment_audio_preview,
                segment_file_output,
                selected_segment_status,
            ],
        )

        director_keyboard_trigger.click(
            fn=apply_director_keyboard_command,
            inputs=[director_table, selected_director_row, director_keyboard_command],
            outputs=[
                director_table,
                selected_director_row,
                segment_audio_preview,
                segment_file_output,
                selected_segment_status,
            ],
            show_progress=False,
            api_name="director_keyboard_command",
        )

        regenerate_segment_btn.click(
            fn=demo.generate_director_segment,
            inputs=[
                language,
                director_table,
                selected_director_row,
                long_reference_wav,
                long_prompt_text,
                long_cfg_value,
                long_normalize,
                long_denoise,
                long_dit_steps,
                use_continuity,
                apply_speed_control,
                apply_prompts_to_tts,
            ],
            outputs=[
                director_table,
                segment_audio_preview,
                segment_file_output,
                selected_segment_status,
            ],
            show_progress=True,
            api_name="regenerate_director_segment",
        )

        director_regenerate_trigger.click(
            fn=demo.generate_director_segment,
            inputs=[
                language,
                director_table,
                selected_director_row,
                long_reference_wav,
                long_prompt_text,
                long_cfg_value,
                long_normalize,
                long_denoise,
                long_dit_steps,
                use_continuity,
                apply_speed_control,
                apply_prompts_to_tts,
            ],
            outputs=[
                director_table,
                segment_audio_preview,
                segment_file_output,
                selected_segment_status,
            ],
            show_progress=True,
            api_name="regenerate_director_segment_from_table",
        )

        zip_segments_btn.click(
            fn=demo.zip_director_segments,
            inputs=[director_table],
            outputs=[segments_zip_output],
            show_progress=True,
            api_name="zip_director_segments",
        )

        merge_segments_btn.click(
            fn=demo.merge_director_segments,
            inputs=[director_table],
            outputs=[long_audio_output, long_generation_status],
            show_progress=True,
            api_name="merge_director_segments",
        )

        generate_long_btn.click(
            fn=demo.generate_longform_audio,
            inputs=[
                language,
                director_table,
                long_reference_wav,
                long_prompt_text,
                long_cfg_value,
                long_normalize,
                long_denoise,
                long_dit_steps,
                rolling_context,
                use_continuity,
                apply_speed_control,
                apply_prompts_to_tts,
            ],
            outputs=[long_audio_output, manifest_output, long_generation_status, director_table],
            show_progress=True,
            api_name="generate_longform",
        )

        language_btn.click(
            fn=_toggle_language,
            inputs=[language],
            outputs=[
                language,
                app_title_markdown,
                language_btn,
                tips_accordion,
                tips_markdown,
                script_module_title,
                long_text,
                long_text_file,
                long_reference_wav,
                long_prompt_text,
                recognize_long_reference_btn,
                max_chars,
                base_style,
                user_style,
                emotion_variation,
                allow_sentence_split,
                build_table_btn,
                director_status,
                director_table,
                long_advanced,
                rolling_context,
                use_continuity,
                apply_speed_control,
                apply_prompts_to_tts,
                generate_long_btn,
                selected_segment_status,
                segment_audio_preview,
                segment_file_output,
                regenerate_segment_btn,
                merge_segments_btn,
                zip_segments_btn,
                segments_zip_output,
                long_audio_output,
                manifest_output,
                long_generation_status,
            ],
        )

    return interface

def run_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.server_name,
        server_port=args.port,
        device=args.device,
    )
