# Clawd — 多端语音助手 (Voice over Hermes)

跨平台语音助手，通过唤醒词触发全链路：`麦克风 → 唤醒词 → ASR (火山引擎 V3) → LLM (Hermes) → TTS (火山引擎 2.0) → 扬声器`

## 支持的平台

| 平台 | 文件 | 说明 |
|------|------|------|
| **Mac** | `platforms/mac/pipeline.py` | 使用 sounddevice 录音，openWakeWord 唤醒，afplay 播放 |
| **树莓派 (Pi)** | `platforms/pi/pipeline.py` | 使用 arecord/aplay ALSA，openWakeWord 唤醒，循环运行 |

## 快速开始

```bash
git clone https://github.com/lzj124/clawd-test.git
cd clawd-test
cp .env.example .env
# 编辑 .env 填入火山引擎 API Key
```

### Mac

```bash
pip install -r requirements.txt
python platforms/mac/pipeline.py
```

### 树莓派 (Debian/ARM Linux)

```bash
# 安装系统依赖
sudo apt-get install -y alsa-utils python3-pip
# 安装 Python 依赖
pip3 install openwakeword websocket-client requests numpy scipy
# 运行
python3 platforms/pi/pipeline.py
```

## 配置

| 变量 | 说明 | 默认 |
|------|------|------|
| `VOLC_TOKEN` | 火山引擎 API Key（新版控制台统一鉴权） | **必填** |
| `ASR_RESOURCE_ID` | ASR 资源 ID | `volc.seedasr.sauc.duration` |
| `HERMES_URL` | Hermes API 地址 | `http://localhost:8642/v1/chat/completions` |
| `LLM_MODEL` | LLM 模型名 | `deepseek-chat` |
| `AUDIO_DEV` | Pi 专用：ALSA 音频设备 | `plughw:2,0` |
| `RECORD_SECS` | 录音秒数 | `5` |

> **鉴权说明：** ASR 和 TTS 均使用火山引擎新版控制台统一鉴权，只需要 `X-Api-Key`（APP Key），不再需要 APP ID。参考文档：[流式语音识别 WebSocket](https://www.volcengine.com/docs/6561/1354869)

## 唤醒词

- **Mac:** `hey jarvis`（英文）
- **Pi:** `alexa`（英文）
- 快捷键 Ctrl+C 可跳过唤醒词直接录音

## 架构

```
clawd-test/
├── .env.example         # 环境变量模板
├── requirements.txt     # Python 依赖（Mac）
├── README.md
├── platforms/
│   ├── mac/
│   │   └── pipeline.py  # Mac 版主脚本
│   └── pi/
│       └── pipeline.py  # 树莓派版主脚本
```
