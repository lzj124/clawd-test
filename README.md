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
| `VOLC_TOKEN` | 火山引擎 API Key（新版控制台统一鉴权） | [火山引擎控制台](https://console.volcengine.com/speech) → 快速入门 |
| `ASR_RESOURCE_ID` | ASR 资源 ID（默认 `volc.seedasr.sauc.duration`） | 可选，使用默认即可 |
| `HERMES_API_KEY` | Hermes API Key | 本地运行的 Hermes gateway 配置 |
| `HERMES_URL` | Hermes API 地址 | 默认 `http://localhost:8642/v1/chat/completions` |

> **鉴权说明：** ASR 和 TTS 均使用火山引擎新版控制台统一鉴权，只需一个 `X-Api-Key`（即 APP Key），不再需要 APP ID。参考文档：[流式语音识别 WebSocket](https://www.volcengine.com/docs/6561/1354869)

## 运行

```bash
python3 pipeline.py
```

唤醒词：**"hey jarvis"** (英文)

检测到唤醒词后录制 5 秒音频 → ASR → LLM → TTS 播放。

快捷键：`Ctrl+C` 跳过唤醒词直接进入录音（调试用）。
