# Clawd — Multi-Device Voice Assistant Test

在 Mac 上通过唤醒词触发的全链路语音助手测试。

**流程：** `麦克风 → 唤醒词 (openWakeWord) → ASR (火山引擎 V3 大模型语音识别) → LLM (Hermes) → TTS (火山引擎 2.0 语音合成) → 扬声器`

## 安装

```bash
pip install -r requirements.txt
# 如果 openwakeword 模型下载失败，手动：
python3 -c "import openwakeword; openwakeword.utils.download_models()"
```

## 配置

```bash
cp .env.example .env
# 编辑 .env 填入密钥
```

| 变量 | 说明 | 获取方式 |
|------|------|----------|
| `VOLC_APPID` | 火山引擎 App ID | [火山引擎控制台](https://console.volcengine.com/speech) |
| `VOLC_TOKEN` | 火山引擎 Access Token（V3 统一鉴权） | 同上 |
| `ASR_RESOURCE_ID` | ASR 资源 ID（默认 `volc.seedasr.sauc.duration`） | 可选，使用默认即可 |
| `HERMES_API_KEY` | Hermes API Key | 本地运行的 Hermes gateway 配置 |
| `HERMES_URL` | Hermes API 地址 | 默认 `http://localhost:8642/v1/chat/completions` |

> **注意：** ASR 使用火山引擎 V3 协议（`/api/v3/sauc/bigmodel_nostream`），统一 `X-Api-Key` 鉴权。旧版 V2（`/api/v2/asr` + `Bearer;token`）已不再使用。

## 运行

```bash
python3 pipeline.py
```

唤醒词：**"hey jarvis"** (英文)

检测到唤醒词后录制 5 秒音频 → ASR → LLM → TTS 播放。

快捷键：`Ctrl+C` 跳过唤醒词直接进入录音（调试用）。
