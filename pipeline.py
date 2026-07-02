#!/usr/bin/env python3
"""
Clawd — 全链路语音助手测试
唤醒词 -> ASR -> LLM -> TTS -> 播放
"""
import os, sys, json, time, base64, struct, uuid, wave, threading
from pathlib import Path

# ── Load .env ──
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

VOLC_APPID = os.environ.get("VOLC_APPID", "")
VOLC_TOKEN = os.environ.get("VOLC_TOKEN", "")
HERMES_KEY = os.environ.get("HERMES_API_KEY", "")
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:8642/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_PROMPT = "你是一个中文语音助手，回答简短直接。"
RECORD_SECS = int(os.environ.get("RECORD_SECS", "5"))

if not VOLC_APPID or not VOLC_TOKEN:
    print("! Missing VOLC_APPID and VOLC_TOKEN, check .env")
    sys.exit(1)


# ── 1. Wake Word Detection ──

def wait_wakeword(timeout=120):
    """Listen until 'hey jarvis' detected"""
    import openwakeword
    from openwakeword.model import Model
    import sounddevice as sd
    import numpy as np

    mdir = Path(openwakeword.__file__).parent / "resources" / "models"
    onx = mdir / "embedding_model.onnx"
    if not onx.exists():
        tfl = mdir / "embedding_model.tflite"
        if tfl.exists():
            import shutil
            shutil.copy(str(tfl), str(onx))

    print("Loading openWakeWord...")
    model = Model(wakeword_models=["hey jarvis"])
    print('Listening for "hey jarvis"... (Ctrl+C to skip)')

    SR, FRAME = 16000, 1280
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME)
    stream.start()

    start = time.time()
    try:
        while time.time() - start < timeout:
            audio, _ = stream.read(FRAME)
            for ww, score in model.predict(audio.flatten()).items():
                if score > 0.5:
                    print(f'! "{ww}" (score={score:.2f})')
                    return True
            if int(time.time() - start) > 0 and int(time.time() - start) % 10 == 0:
                print(f"  listening... ({int(time.time()-start)}s)")
    except KeyboardInterrupt:
        print("\nskip wake word")
        return True
    finally:
        stream.stop()
        stream.close()

    print("timeout")
    return False


# ── 2. Record ──

def record(secs=5):
    """Record mic, return PCM16 mono 16kHz bytes"""
    import sounddevice as sd
    import numpy as np

    print(f"Recording {secs}s...")
    audio = sd.rec(int(16000 * secs), samplerate=16000, channels=1, dtype="int16")
    for i in range(secs):
        time.sleep(1)
        print(f"  {i+1}/{secs}")
    sd.wait()
    print("Done")
    return audio.flatten().tobytes()


# ── 3. ASR (Volcengine V2 WebSocket) ──

def asr(pcm):
    import websocket

    ws = websocket.WebSocket()
    ws.settimeout(15)
    ws.connect("wss://openspeech.bytedance.com/api/v2/asr",
               header={"Authorization": f"Bearer;{VOLC_TOKEN}"})

    def hdr(t, f):
        return bytes([0x11, (t << 4) | f, 0x10, 0x00])

    cfg = json.dumps({
        "app": {"appid": VOLC_APPID, "token": VOLC_TOKEN, "cluster": "volcengine_streaming_common"},
        "user": {"uid": "clawd-test"},
        "audio": {"format": "raw", "rate": 16000, "bits": 16, "channel": 1,
                  "language": "zh-CN"},
        "request": {"reqid": str(uuid.uuid4()), "nbest": 1, "show_utterances": True},
    }).encode()
    ws.send(hdr(1, 0) + struct.pack(">I", len(cfg)) + cfg)
    ws.send(hdr(2, 2) + struct.pack(">I", len(pcm)) + pcm)

    text = ""
    while True:
        msg = ws.recv()
        if not isinstance(msg, bytes) or len(msg) < 8:
            continue
        pl = msg[8:8 + struct.unpack(">I", msg[4:8])[0]]
        t = (msg[1] >> 4) & 0xF
        if t == 9:
            r = json.loads(pl)
            if r.get("code") == 0:
                for seg in r.get("result", []):
                    if seg.get("text"):
                        text = seg["text"]
            if r.get("payload_msg", r).get("is_final"):
                break
        elif t == 0xF:
            break
    ws.close()

    if text:
        print(f'ASR: "{text}"')
    else:
        print("ASR: no result")
    return text


# ── 4. LLM (Hermes API) ──

def chat(text):
    import requests

    print("LLM...")
    hdrs = {"Content-Type": "application/json"}
    if HERMES_KEY:
        hdrs["Authorization"] = f"Bearer {HERMES_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    try:
        resp = requests.post(HERMES_URL, json=payload, headers=hdrs, timeout=30)
        reply = resp.json()["choices"][0]["message"]["content"]
        print(f"LLM: {reply[:100]}")
        return reply
    except Exception as e:
        print(f"LLM failed: {e}")
        return None


# ── 5. TTS (Volcengine V3) -> play ──

def speak(text):
    import requests

    print("TTS...")
    hdrs = {
        "Content-Type": "application/json",
        "X-Api-Key": VOLC_TOKEN,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Request-Id": str(uuid.uuid4()),
    }
    payload = {
        "user": {"uid": "clawd-test"},
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
        print(f"TTS HTTP {resp.status_code}")
        return False

    pcm = bytearray()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get("code") in (0, 20000000) and c.get("data"):
                pcm.extend(base64.b64decode(c["data"]))
        except json.JSONDecodeError:
            continue

    if not pcm:
        print("TTS: no audio")
        return False

    out = Path.home() / "Desktop" / "clawd-test" / "output.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(bytes(pcm))
    print(f"Saved {out} ({len(pcm)}B)")

    import subprocess
    subprocess.run(["afplay", str(out)])
    print("Played")
    return True


# ── Main ──

def main():
    print("=" * 50)
    print("  Clawd Pipeline Test")
    print("=" * 50)

    if not wait_wakeword():
        sys.exit(1)

    audio = record(RECORD_SECS)
    text = asr(audio)
    if not text:
        sys.exit(1)

    reply = chat(text)
    if not reply:
        sys.exit(1)

    speak(reply)
    print(f"\nOK: '{text}' -> '{reply}'")


if __name__ == "__main__":
    main()
