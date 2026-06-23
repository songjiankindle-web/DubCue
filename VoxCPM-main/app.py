import os
import re
import sys
import logging
import numpy as np
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.longform import (
    build_director_segments,
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
        "single_tab": "单句生成",
        "long_tab": "VoxDirector 长文本配音",
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
        "reference_transcript": "参考音频文本（可选，用于增强连续克隆）",
        "max_chars": "每段最大字数",
        "style": "整体配音风格",
        "style_notes": "全局风格补充",
        "style_notes_placeholder": "如：冷静的自然纪录片，语速缓慢，克制",
        "variation": "情绪变化幅度",
        "allow_split": "允许对超长句按自然停顿二次切分",
        "build_table": "生成导演表",
        "status": "状态",
        "table": "Director Table / 导演表",
        "table_headers": ["序号", "文本", "情绪", "语速", "提示词", "停顿 ms", "状态"],
        "long_settings": "长文本生成设置",
        "rolling": "滚动连续上下文段数",
        "continuity": "使用滚动 prompt cache 保持音色连续",
        "speed_control": "对每段应用温和语速修正",
        "apply_prompts": "实验性：把每段提示词送入 TTS（可能被读出来，默认关闭）",
        "generate_long": "生成完整长文本音频",
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
        "single_tab": "Single Clip",
        "long_tab": "VoxDirector Long-form",
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
        "reference_transcript": "Reference transcript (optional, improves continuation cloning)",
        "max_chars": "Maximum characters per segment",
        "style": "Overall narration style",
        "style_notes": "Global style notes",
        "style_notes_placeholder": "e.g. calm nature documentary, slow and restrained",
        "variation": "Emotion variation",
        "allow_split": "Allow splitting very long sentences at natural pauses",
        "build_table": "Build Director Table",
        "status": "Status",
        "table": "Director Table",
        "table_headers": ["Index", "Text", "Emotion", "Speed", "Prompt", "Pause ms", "Status"],
        "long_settings": "Long-form generation settings",
        "rolling": "Rolling continuity segments",
        "continuity": "Keep voice continuous with rolling prompt cache",
        "speed_control": "Apply gentle speed correction after each segment",
        "apply_prompts": "Experimental: send segment prompts into TTS (may be spoken, off by default)",
        "generate_long": "Generate Long-form Audio",
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
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
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
        return segments_to_rows(segments), ui_text(language, "ready").format(count=len(segments))

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
        segments = rows_to_segments(director_rows)
        if not segments:
            raise ValueError("The director table is empty.")

        current_model = self.get_or_load_voxcpm()
        output_root = Path.cwd() / "outputs" / "voxdirector"

        def _progress_callback(current: int, total: int, stage: str, text: str):
            stage_labels = {
                "generating": "Generating",
                "saved": "Saved",
                "assembling": "Assembling final audio",
                "done": "Done",
            }
            preview = (text or "").strip().replace("\n", " ")
            if len(preview) > 34:
                preview = preview[:34] + "..."
            if stage == "generating":
                desc = ui_text(language, "progress_generating").format(
                    current=current,
                    total=total,
                    preview=preview,
                )
            elif stage == "saved":
                desc = ui_text(language, "progress_saved").format(current=current, total=total)
            elif stage == "assembling":
                desc = ui_text(language, "progress_assembling")
            elif stage == "done":
                desc = ui_text(language, "progress_done")
            else:
                desc = stage_labels.get(stage, stage)
            progress(current / max(total, 1), desc=desc)

        progress(0, desc=ui_text(language, "preparing").format(count=len(segments)))
        result = synthesize_longform(
            model=current_model,
            segments=segments,
            output_root=output_root,
            reference_wav_path=reference_wav_path_input if reference_wav_path_input else None,
            prompt_text=(prompt_text or "").strip(),
            cfg_value=cfg_value_input,
            inference_timesteps=int(inference_timesteps),
            normalize=do_normalize,
            denoise=denoise,
            rolling_context_segments=int(rolling_context_segments),
            use_continuity=use_continuity,
            apply_speed_control=apply_speed_control,
            apply_prompts_to_tts=apply_prompts_to_tts,
            progress_callback=_progress_callback,
        )
        message = ui_text(language, "done").format(
            count=result.segment_count,
            duration=result.duration_seconds,
            manifest=result.manifest_path,
        )
        return result.output_path, result.manifest_path, message


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

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

    def _apply_language(language: str):
        t = UI_TEXT[language if language in UI_TEXT else "zh"]
        return [
            gr.update(value=t["usage"]),
            gr.update(label=t["reference_audio"]),
            gr.update(label=t["ultimate_mode"], info=t["ultimate_info"]),
            gr.update(label=t["prompt_text"], placeholder=t["prompt_placeholder"]),
            gr.update(label=t["control"], placeholder=t["control_placeholder"]),
            gr.update(label=t["target_text"]),
            gr.update(label=t["advanced"]),
            gr.update(label=t["denoise"], info=t["denoise_info"]),
            gr.update(label=t["normalize"], info=t["normalize_info"]),
            gr.update(label=t["cfg"], info=t["cfg_info"]),
            gr.update(label=t["dit"], info=t["dit_info"]),
            gr.update(value=t["generate"]),
            gr.update(label=t["generated_audio"]),
            gr.update(value=t["examples"]),
            gr.update(label=t["long_script"], placeholder=t["long_script_placeholder"]),
            gr.update(label=t["text_file"]),
            gr.update(label=t["reference_voice"]),
            gr.update(label=t["reference_transcript"]),
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
            gr.update(label=t["final_audio"]),
            gr.update(label=t["manifest"]),
            gr.update(label=t["generation_status"]),
        ]

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )
        language = gr.Radio(
            choices=[("中文", "zh"), ("English", "en")],
            value="zh",
            label="界面语言 / Interface Language",
        )

        with gr.Tabs():
            with gr.Tab("单句生成 / Single Clip"):
                usage_markdown = gr.Markdown(ui_text("zh", "usage"))

                with gr.Row():
                    with gr.Column():
                        reference_wav = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label=ui_text("zh", "reference_audio"),
                        )
                        show_prompt_text = gr.Checkbox(
                            value=False,
                            label=ui_text("zh", "ultimate_mode"),
                            info=ui_text("zh", "ultimate_info"),
                            elem_classes=["switch-toggle"],
                        )
                        prompt_text = gr.Textbox(
                            value="",
                            label=ui_text("zh", "prompt_text"),
                            placeholder=ui_text("zh", "prompt_placeholder"),
                            lines=2,
                            visible=False,
                        )
                        control_instruction = gr.Textbox(
                            value="",
                            label=ui_text("zh", "control"),
                            placeholder=ui_text("zh", "control_placeholder"),
                            lines=2,
                        )
                        text = gr.Textbox(
                            value=DEFAULT_TARGET_TEXT,
                            label=ui_text("zh", "target_text"),
                            lines=3,
                        )

                        with gr.Accordion(ui_text("zh", "advanced"), open=False) as single_advanced:
                            DoDenoisePromptAudio = gr.Checkbox(
                                value=False,
                                label=ui_text("zh", "denoise"),
                                elem_classes=["switch-toggle"],
                                info=ui_text("zh", "denoise_info"),
                            )
                            DoNormalizeText = gr.Checkbox(
                                value=False,
                                label=ui_text("zh", "normalize"),
                                elem_classes=["switch-toggle"],
                                info=ui_text("zh", "normalize_info"),
                            )
                            cfg_value = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=2.0,
                                step=0.1,
                                label=ui_text("zh", "cfg"),
                                info=ui_text("zh", "cfg_info"),
                            )
                            dit_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label=ui_text("zh", "dit"),
                                info=ui_text("zh", "dit_info"),
                            )

                        run_btn = gr.Button(ui_text("zh", "generate"), variant="primary", size="lg")

                    with gr.Column():
                        audio_output = gr.Audio(label=ui_text("zh", "generated_audio"))
                        examples_markdown = gr.Markdown(ui_text("zh", "examples"))

                show_prompt_text.change(
                    fn=_on_toggle_instant,
                    inputs=[show_prompt_text],
                    outputs=[prompt_text, control_instruction],
                ).then(
                    fn=_run_asr_if_needed,
                    inputs=[show_prompt_text, reference_wav],
                    outputs=[prompt_text],
                )

                run_btn.click(
                    fn=_generate,
                    inputs=[
                        text,
                        control_instruction,
                        reference_wav,
                        show_prompt_text,
                        prompt_text,
                        cfg_value,
                        DoNormalizeText,
                        DoDenoisePromptAudio,
                        dit_steps,
                    ],
                    outputs=[audio_output],
                    show_progress=True,
                    api_name="generate",
                )

            with gr.Tab("VoxDirector 长文本配音 / Long-form"):
                with gr.Row():
                    with gr.Column(scale=1):
                        long_text = gr.Textbox(
                            label=ui_text("zh", "long_script"),
                            placeholder=ui_text("zh", "long_script_placeholder"),
                            lines=12,
                        )
                        long_text_file = gr.File(
                            label=ui_text("zh", "text_file"),
                            file_types=[".txt", ".md"],
                        )
                        long_reference_wav = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label=ui_text("zh", "reference_voice"),
                        )
                        long_prompt_text = gr.Textbox(
                            value="",
                            label=ui_text("zh", "reference_transcript"),
                            lines=2,
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
                            label=ui_text("zh", "status"),
                            value="",
                            interactive=False,
                        )

                director_table = gr.Dataframe(
                    headers=[
                        *ui_text("zh", "table_headers"),
                    ],
                    datatype=["number", "str", "str", "str", "str", "number", "str"],
                    interactive=True,
                    wrap=True,
                    label=ui_text("zh", "table"),
                )

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
                                value=3,
                                step=1,
                                label=ui_text("zh", "rolling"),
                            )
                            use_continuity = gr.Checkbox(
                                value=True,
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

                    with gr.Column():
                        long_audio_output = gr.Audio(label=ui_text("zh", "final_audio"))
                        manifest_output = gr.File(label=ui_text("zh", "manifest"))
                        long_generation_status = gr.Textbox(
                            label=ui_text("zh", "generation_status"),
                            value="",
                            interactive=False,
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
                    outputs=[long_audio_output, manifest_output, long_generation_status],
                    show_progress=True,
                    api_name="generate_longform",
                )

        language.change(
            fn=_apply_language,
            inputs=[language],
            outputs=[
                usage_markdown,
                reference_wav,
                show_prompt_text,
                prompt_text,
                control_instruction,
                text,
                single_advanced,
                DoDenoisePromptAudio,
                DoNormalizeText,
                cfg_value,
                dit_steps,
                run_btn,
                audio_output,
                examples_markdown,
                long_text,
                long_text_file,
                long_reference_wav,
                long_prompt_text,
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
