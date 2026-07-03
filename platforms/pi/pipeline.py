#!/usr/bin/env python3
"""ClawVoice Pi — 全链路语音助手 (树莓派/Debian)"""
import os, sys, json, time, base64, struct, uuid, wave, subprocess
from pathlib import Path

# ── 配置（优先读取 .env） ──
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("VOLC_TOKEN", "")
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:8642/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_PROMPT = "你是一个中文语音助手，回答简短直接。"
ASR_RESOURCE_ID = os.environ.get("ASR_RESOURCE_ID", "volc.seedasr.sauc.duration")
AUDIO_DEV = os.environ.get("AUDIO_DEV", "plughw:2,0")
RECORD_SECS = int(os.environ.get("RECORD_SECS", "5"))

if not API_KEY:
    print("! Missing VOLC_TOKEN, check .env")
    sys.exit(1)

# ── 音频工具 ──

def play_wav(path):
    subprocess.run(["aplay", "-D", AUDIO_DEV, str(path)], capture_output=True)

def record_wav(path, secs=RECORD_SECS):
    print(f"  录音 {secs} 秒...")
    subprocess.run(["arecord", "-D", AUDIO_DEV, "-d", str(secs),
                    "-r", "16000", "-f", "S16_LE", "-c", "1", str(path)],
                   capture_output=True)
    print("  录音完成")

def wav_to_pcm(wav_path):
    with open(wav_path, "rb") as f:
        data = f.read()
    if data[:4] == b'RIFF':
        return data[44:]
    return data

# ── 1. 唤醒词 ──

def wait_wakeword():
    import openwakeword
    from openwakeword.model import Model
    import numpy as np

    mdir = Path(openwakeword.__file__).parent / "resources" / "models"
    has_models = any(mdir.glob("*.onnx")) or any(mdir.glob("*.tflite"))
    if not has_models:
        print("下载唤醒词模型...")
        openwakeword.utils.download_models()

    onx = mdir / "embedding_model.onnx"
    if not onx.exists():
        tfl = mdir / "embedding_model.tflite"
        if tfl.exists():
            import shutil
            shutil.copy(str(tfl), str(onx))

    print("加载唤醒词模型...")
    model = Model(wakeword_models=["alexa"])
    print('🎤 监听 "alexa"...')

    SR, FRAME = 16000, 1280
    proc = subprocess.Popen(
        ["arecord", "-D", AUDIO_DEV, "-r", str(SR),
         "-f", "S16_LE", "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            raw = proc.stdout.read(FRAME * 2)
            if len(raw) < FRAME * 2:
                break
            audio = np.frombuffer(raw, dtype=np.int16)
            for ww, score in model.predict(audio).items():
                if score > 0.5:
                    print(f'🔔 "{ww}"! (score={score:.2f})')
                    return True
    except KeyboardInterrupt:
        return False
    finally:
        proc.terminate()
        proc.wait()
    return False

# ── 2. 提示音 ──

def ensure_prompt_audio():
    prompt_path = Path("/tmp/voice_prompt.wav")
    if prompt_path.exists():
        return prompt_path
    print("生成提示音...")
    pcm = tts_synthesize("在呢")
    if pcm:
        with wave.open(str(prompt_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(bytes(pcm))
        return prompt_path
    return None

# ── 3. ASR (V3 WebSocket — 新版 统一鉴权) ──

def asr_transcribe(pcm):
    import websocket

    ws = websocket.WebSocket()
    ws.settimeout(20)
    ws.connect("wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
               header={
                   "X-Api-Key": API_KEY,
                   "X-Api-Resource-Id": ASR_RESOURCE_ID,
                   "X-Api-Request-Id": str(uuid.uuid4()),
                   "X-Api-Sequence": "-1",
               })

    def build_header(msg_type, flags, serialization=1, compression=0):
        return bytes([0x11, (msg_type << 4) | flags,
                      (serialization << 4) | compression, 0x00])

    reqid = str(uuid.uuid4())
    config = json.dumps({
        "user": {"uid": "clawd-pi"},
        "audio": {"format": "pcm", "rate": 16000, "bits": 16, "channel": 1},
        "request": {"reqid": reqid, "nbest": 1, "show_utterances": True},
    }).encode()
    ws.send(build_header(0x01, 0x00) + struct.pack(">I", len(config)) + config,
            opcode=websocket.ABNF.OPCODE_BINARY)

    chunk_size = 3200
    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset:offset+chunk_size]
        is_last = (offset + chunk_size >= len(pcm))
        flags = 0x02 if is_last else 0x00
        ws.send(build_header(0x02, flags) +
                struct.pack(">I", len(chunk)) + chunk,
                opcode=websocket.ABNF.OPCODE_BINARY)
        time.sleep(0.05)

    text = ""
    msg_count = 0
    while True:
        try:
            msg = ws.recv()
        except Exception as e:
            print(f"  ASR recv: {e}")
            break
        msg_count += 1
        if not isinstance(msg, bytes) or len(msg) < 8:
            continue
        payload_size = struct.unpack(">I", msg[4:8])[0]
        payload = msg[8:8+payload_size]
        msg_type = (msg[1] >> 4) & 0x0F
        if msg_type != 0x09:
            continue
        if len(payload) < 8:
            continue
        pdata = payload[4:]
        try:
            r = json.loads(pdata)
            if r.get("code") == 0 and "result" in r:
                for seg in r["result"]:
                    t = seg.get("text", "")
                    if t:
                        text = t
            pm = r.get("payload_msg", r)
            if pm.get("is_final"):
                break
        except:
            pass
        if msg_count > 60:
            break
    ws.close()
    if text:
        print(f'📝 "{text}"')
    else:
        print("⚠️ ASR 无结果")
    return text

# ── 4. LLM (Hermes /v1/runs — 带记忆) ──

def llm_chat(text):
    import requests
    print("🤖 LLM...")
    try:
        # Create run
        url = HERMES_URL.replace("/chat/completions", "/v1/runs")
        resp = requests.post(
            url,
            json={"input": text, "instructions": LLM_PROMPT},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code not in (200, 202):
            print(f"  Run HTTP {resp.status_code}")
            return None

        run_id = resp.json().get("run_id") or resp.json().get("id", "")
        if not run_id:
            print(f"  No run_id: {resp.text[:200]}")
            return None

        # Stream events
        ev_url = HERMES_URL.replace("/chat/completions", f"/v1/runs/{run_id}/events")
        ev_resp = requests.get(ev_url, headers={"Content-Type": "application/json"}, timeout=120, stream=True)
        if ev_resp.status_code != 200:
            print(f"  Events HTTP {ev_resp.status_code}")
            return None

        reply = ""
        for line in ev_resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            event = ev.get("event", "")

            if event == "approval.request":
                # Auto-approve
                approve_url = HERMES_URL.replace(
                    "/chat/completions", f"/v1/runs/{run_id}/approval"
                )
                try:
                    requests.post(approve_url, json={"choice": "always"},
                                  headers={"Content-Type": "application/json"}, timeout=5)
                except Exception:
                    pass
                continue
            elif event == "message.delta":
                d = ev.get("delta", "")
                if d:
                    reply += d
            elif event == "run.completed":
                out = ev.get("output", "")
                if out and not reply:
                    reply = out
                break
            elif event == "run.failed":
                print(f"  Run failed: {ev.get('error', 'unknown')}")
                break

        reply = reply.strip()
        if reply:
            print(f'💬 {reply[:100]}')
        else:
            print("💬 (empty)")
        return reply
    except Exception as e:
        print(f"❌ LLM: {e}")
        return None

# ── 5. TTS (V3, x-api-key) ──

def tts_synthesize(text):
    import requests
    hdrs = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Request-Id": str(uuid.uuid4()),
    }
    payload = {
        "user": {"uid": "clawd-pi"},
        "namespace": "BidirectionalTTS",
        "req_params": {
            "text": text,
            "speaker": "zh_female_vv_uranus_bigtts",
            "audio_params": {"format": "pcm", "sample_rate": 8000},
        },
    }
    resp = requests.post(
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
        json=payload, headers=hdrs, timeout=30, stream=True,
    )
    if resp.status_code != 200:
        return None
    pcm = bytearray()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get("data"):
                pcm.extend(base64.b64decode(c["data"]))
        except:
            pass
    return bytes(pcm) if pcm else None

def tts_play(text):
    print(f"🔊 TTS: {text[:50]}")
    pcm = tts_synthesize(text)
    if not pcm:
        print("❌ TTS 失败")
        return False
    out = Path("/tmp/tts_reply.wav")
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    play_wav(out)
    print("🔊 播放完成")
    return True

# ── Main Loop ──

def main():
    print("=" * 50)
    print("  ClawVoice Pi — 全链路 (V3 API)")
    print("  唤醒词 -> 在呢 -> 录制 -> ASR -> LLM -> TTS")
    print("=" * 50)

    prompt = ensure_prompt_audio()

    while True:
        if not wait_wakeword():
            break
        if prompt:
            print("📢 在呢!")
            play_wav(prompt)
        wav_path = Path("/tmp/voice_input.wav")
        record_wav(wav_path)
        pcm = wav_to_pcm(wav_path)
        text = asr_transcribe(pcm)
        if not text:
            tts_play("抱歉，没有听清，请再说一遍")
            continue
        reply = llm_chat(text)
        if not reply:
            tts_play("抱歉，我出错了")
            continue
        tts_play(reply)
        print("-" * 50)
    print("再见！")

if __name__ == "__main__":
    main()
