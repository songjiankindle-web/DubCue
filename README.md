# VoxCPM Studio

[English](README_EN.md) | 简体中文

> 一个面向普通用户的 VoxCPM 本地安装与图形界面外壳。

**VoxCPM Studio 不训练、不修改、也不拥有 VoxCPM 模型。**  
它只是为优秀的开源项目 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 提供更容易安装、启动和使用的桌面友好外壳。

## 请先访问原项目

VoxCPM 的模型、推理代码、技术成果与核心能力均来自 OpenBMB / ModelBest 团队：

- **官方 GitHub：** [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **官方在线体验：** [Hugging Face Space](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo)
- **官方国内体验：** [voxcpm.modelbest.cn](https://voxcpm.modelbest.cn/)
- **官方文档：** [VoxCPM Documentation](https://voxcpm.readthedocs.io/zh-cn/latest/)
- **模型权重：** [Hugging Face](https://huggingface.co/openbmb/VoxCPM2) / [ModelScope](https://modelscope.cn/models/OpenBMB/VoxCPM2)

请优先阅读、关注并支持原项目。本仓库不是 OpenBMB 或 ModelBest 的官方仓库，也不代表原项目团队。

## 这个项目做了什么

VoxCPM Studio 将 VoxCPM 包装成适合非开发者使用的本地应用：

- 提供 macOS 和 Windows 安装流程
- 提供双击启动入口
- 自动创建独立 Python 虚拟环境
- 使用本地浏览器图形界面，无需手写命令
- 支持声音设计、可控声音克隆和极致克隆
- 模型安装完成后在本机推理
- 不调用 Codex、ChatGPT 或其他 Agent
- 本地生成不消耗 API token

图形界面本身基于 VoxCPM 原项目提供的 Gradio Demo 进行桌面化包装与启动适配。

## 界面预览

![VoxCPM Studio GUI](assets/voxcpm-studio-gui.png)

## 下载

由于安装包包含本地模型权重，体积约为 4–6 GB，不直接存放在 Git 仓库中。

安装包将由维护者单独上传。下载位置和校验信息见：

**[下载与安装说明](DOWNLOADS.md)**

预留版本：

- macOS Apple Silicon 安装包
- Windows 10/11 x64 安装包

## 本地与隐私

安装完成后，普通语音生成和声音克隆在本机运行，不经过 Codex、ChatGPT 或第三方 Agent，也不会消耗 Agent/API token。

以下场景可能需要联网：

- 首次安装 Python 依赖
- 安装 FFmpeg
- 首次使用某些附加模型，例如自动语音识别
- 用户主动使用官方在线服务

请勿使用声音克隆功能冒充他人、实施欺诈或传播虚假信息。建议明确标注 AI 生成内容，并仅在获得授权的情况下使用他人声音。

## 系统要求

### macOS

- Apple Silicon Mac：M1 / M2 / M3 / M4
- Python 3.10、3.11 或 3.12
- 推荐 Python 3.11

### Windows

- 64 位 Windows 10/11
- Python 3.10、3.11 或 3.12
- 推荐 Python 3.11
- 推荐 NVIDIA 显卡；CPU 模式通常较慢

Python 3.13/3.14 暂不作为推荐环境，因为 VoxCPM 的完整依赖链尚未在这些版本上稳定验证。

## 致谢

衷心感谢：

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 项目及其所有贡献者
- OpenBMB 与 ModelBest 团队
- PyTorch、Gradio、Hugging Face、ModelScope、FFmpeg 及相关开源社区

如果 VoxCPM 对你有帮助，请前往[原项目](https://github.com/OpenBMB/VoxCPM)点 Star、阅读许可证、技术文档与最新发布信息。

## 许可证与归属

本仓库中的外围说明、安装脚本和启动包装代码采用 [MIT License](LICENSE)。

VoxCPM 源码、模型权重、名称、标识及相关资产仍遵循其各自原始许可证与归属。本仓库的 MIT License 不会覆盖或替代 VoxCPM 原项目及模型的许可证。

