# Clawd — 多端持续语音助手 (Voice over Hermes)

跨平台持续语音助手：**唤醒词 → 录音 → ASR (火山引擎 V3) → LLM (Hermes /v1/runs) → TTS (火山引擎 2.0) → 循环**

## 快速开始

```bash
git clone https://github.com/lzj124/clawd-test.git
cd clawd-test
cp .env.example .env
# 编辑 .env 填入 VOLC_TOKEN
```

### Mac

```bash
pip install -r requirements.txt
python platforms/mac/run.py
```

### 树莓派 (Debian/ARM)

```bash
sudo apt-get install -y alsa-utils python3-pip
pip3 install openwakeword websocket-client requests numpy scipy
python3 platforms/pi/run.py
```

## 配置

| 变量 | 说明 | 默认 |
|------|------|------|
| `VOLC_TOKEN` | 火山引擎 API Key（新版统一鉴权） | **必填** |
| `API_SERVER_KEY` | Hermes API 本地鉴权 | `clawd` |
| `RECORD_SECS` | 录音秒数 | `5` |
| `AUDIO_DEV` | Pi 专用：ALSA 设备 | `plughw:2,0` |

唤醒词：**"alexa"**。Ctrl+C 退出。

## 架构

```
clawd/
├── __init__.py        ← 共享核心：ASR / LLM / TTS / Hermes 自启
platforms/
├── mac/run.py         ← Mac 版 (sounddevice + afplay)
└── pi/run.py          ← Pi 版 (arecord + aplay)
```
