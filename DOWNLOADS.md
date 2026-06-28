# Downloads / 下载

## 历史版本：VoxDirector v0.2.2

这是 DubCue 更名前的历史版本。VoxDirector 是 VoxCPM Easy Launcher 的大版本升级，新增长文本配音、智能断句、Director Table 手动编辑、自动情绪提示词、连续音色拼接、界面语言切换和长任务实时进度显示。v0.2.2 修复了长文本模式可能把提示词读出来、生成超长静音尾巴、坏段继续拼接的问题。

### macOS Apple Silicon

- **File / 文件：** `VoxDirector-macOS-Lite.zip`
- **Download / 下载：** [GitHub Release](https://github.com/songjiankindle-web/DubCue/releases/download/v0.2.2/VoxDirector-macOS-Lite.zip)
- **SHA-256：** `56a4cbe45011f20911630c23c713e629754f1ccdf4008cb981f02c3f530941e1`
- **Target / 适用：** Apple Silicon M1 / M2 / M3 / M4

安装方法：

1. 下载并解压。
2. 右键 `Install.command`，选择“打开”。
3. 安装完成后双击 `~/Applications/VoxDirector/Start VoxDirector.command`。

### Windows x64

- **File / 文件：** `VoxDirector-Windows-Lite.zip`
- **Download / 下载：** [GitHub Release](https://github.com/songjiankindle-web/DubCue/releases/download/v0.2.2/VoxDirector-Windows-Lite.zip)
- **SHA-256：** `e67fa6e2af31d2b7de249a3e1fb3d14b3a8d195b5d34c56d9ad871b872686107`
- **Target / 适用：** 64-bit Windows 10/11

安装方法：

1. 下载并解压，不要直接在压缩包内部运行。
2. 双击 `Install.bat`。
3. 安装完成后双击 `%LOCALAPPDATA%\VoxDirector\Start VoxDirector.bat`。

## 历史版本：VoxCPM Easy Launcher v0.1.0-lite

旧版轻量包继续保留，适合只需要单句生成和原 Easy Launcher 工作流的用户。

## 完整版：已内置本地模型

> [!IMPORTANT]
> **不想在安装时等待约 4.6GB 模型下载，请选择完整版。**
>
> **百度网盘：[点击下载 macOS / Windows 完整版](https://pan.baidu.com/s/1pRR-aB4NA1Dby3Goh6oA_Q?pwd=sjnb)**
>
> **提取码：`sjnb`**

完整版已经包含 VoxCPM2 本地模型，不需要再次下载主模型。首次安装 Python 依赖、FFmpeg 或使用可选附加模型时仍可能需要联网。

## 轻量版：GitHub Release

轻量版约 3MB，不包含任何模型权重。安装器会在用户电脑上自动下载依赖、FFmpeg 和约 4.6GB 的 VoxCPM2 模型。

## macOS Apple Silicon

- **File / 文件：** `VoxCPM-Easy-Launcher-macOS-Lite.zip`
- **Download / 下载：** [GitHub Release](https://github.com/songjiankindle-web/DubCue/releases/download/v0.1.0-lite/VoxCPM-Easy-Launcher-macOS-Lite.zip)
- **SHA-256：** `ad1a9d3048384ef4589ebee27f6cc167dba8a2f78767fa62197c1e5367b41453`
- **Target / 适用：** Apple Silicon M1 / M2 / M3 / M4

安装方法：

1. 下载并解压。
2. 右键 `Install.command`，选择“打开”。
3. 安装完成后双击 `Start VoxCPM.command`。

## Windows x64

- **File / 文件：** `VoxCPM-Easy-Launcher-Windows-Lite.zip`
- **Download / 下载：** [GitHub Release](https://github.com/songjiankindle-web/DubCue/releases/download/v0.1.0-lite/VoxCPM-Easy-Launcher-Windows-Lite.zip)
- **SHA-256：** `2a6b4a9963f6b6dec4b2477f70db6fe038480d24b260729f9397b82fdc311112`
- **Target / 适用：** 64-bit Windows 10/11

安装方法：

1. 下载并解压，不要直接在压缩包内部运行。
2. 双击 `Install.bat`。
3. 安装完成后双击 `%LOCALAPPDATA%\VoxCPM-Studio\Start VoxCPM.bat`。

## Important / 重要说明

- 模型及核心代码来自 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)。
- 轻量版会自动下载模型；完整版已经包含本地模型。
- 下载后建议核对 SHA-256。
- 安装 Python 依赖和 FFmpeg 时可能需要联网。
- 本地生成不使用 Codex、ChatGPT 或 Agent token。
- 请遵守原项目许可证，并负责任地使用声音克隆功能。
