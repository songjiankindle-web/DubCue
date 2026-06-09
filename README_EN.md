# VoxCPM Studio

[简体中文](README.md) | English

> A desktop-friendly local installer and GUI wrapper for VoxCPM.

**VoxCPM Studio does not train, modify, or own the VoxCPM model.** It makes the excellent open-source [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) project easier for non-developers to install, launch, and use.

## Visit The Original Project

The model, inference code, research, and core capabilities belong to the OpenBMB / ModelBest team:

- **Official GitHub:** [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Official online demo:** [Hugging Face Space](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo)
- **Official China demo:** [voxcpm.modelbest.cn](https://voxcpm.modelbest.cn/)
- **Documentation:** [VoxCPM Documentation](https://voxcpm.readthedocs.io/en/latest/)
- **Model weights:** [Hugging Face](https://huggingface.co/openbmb/VoxCPM2) / [ModelScope](https://modelscope.cn/models/OpenBMB/VoxCPM2)

Please read, follow, and support the original project first. This repository is unofficial and is not affiliated with or endorsed by OpenBMB or ModelBest.

## What This Wrapper Adds

- Guided macOS and Windows installation
- Double-click launchers
- An isolated Python virtual environment
- A browser-based local GUI instead of terminal commands
- Voice design, controllable cloning, and transcript-guided cloning
- Local inference after installation
- No Codex, ChatGPT, or agent dependency
- No API-token usage for local generation

The GUI is a desktop-oriented packaging and launch adaptation of the Gradio demo provided by the VoxCPM project.

## Interface

![VoxCPM Studio GUI](assets/voxcpm-studio-gui.png)

## Downloads

Installers contain local model weights and are roughly 4–6 GB, so they are not committed to this Git repository.

See **[Downloads and installation](DOWNLOADS.md)** for release links and checksums.

Planned packages:

- macOS Apple Silicon
- Windows 10/11 x64

## Local Use And Privacy

After installation, normal synthesis and voice cloning run locally. They do not use Codex, ChatGPT, or another agent, and do not consume API tokens.

Internet access may still be needed for initial Python dependencies, FFmpeg, optional ASR models, or official online services selected by the user.

Do not use voice cloning for impersonation, fraud, or misinformation. Obtain permission before using another person's voice and clearly label AI-generated content.

## Requirements

- macOS Apple Silicon or 64-bit Windows 10/11
- Python 3.10–3.12; Python 3.11 recommended
- NVIDIA GPU recommended on Windows; CPU inference can be very slow

Python 3.13/3.14 are not recommended until the complete VoxCPM dependency chain has been reliably validated on them.

## Acknowledgements

Special thanks to:

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) and all contributors
- The OpenBMB and ModelBest teams
- PyTorch, Gradio, Hugging Face, ModelScope, FFmpeg, and their communities

If VoxCPM helps you, please star and support the [original repository](https://github.com/OpenBMB/VoxCPM).

## License And Attribution

The wrapper documentation, installer scripts, and launcher code in this repository use the [MIT License](LICENSE).

VoxCPM source code, model weights, names, logos, and related assets remain subject to their original licenses and attribution. This repository's MIT License does not replace them.

