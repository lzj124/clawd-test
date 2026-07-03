#!/usr/bin/env python3
"""
Clawd — 全链路语音助手测试
唤醒词 -> ASR -> LLM -> TTS -> 播放
"""
import os, sys, json, time, base64, struct, uuid, wave, threading, socket
from pathlib import Path

# ── Load .env ──
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

VOLC_TOKEN = os.environ.get("VOLC_TOKEN", "")
HERMES_KEY = os.environ.get("HERMES_API_KEY", "")
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:8642/v1/chat/completions")
LLM_PROMPT = "你是一个中文语音助手，回答简短直接。"
RECORD_SECS = int(os.environ.get("RECORD_SECS", "5"))

if not VOLC_TOKEN:
    print("! Missing VOLC_TOKEN, check .env")
    sys.exit(1)

API_SERVER_KEY = os.environ.get("API_SERVER_KEY", "clawd")


# ── 0. Auto-start Hermes gateway ──

def ensure_hermes():
    """Start Hermes gateway if not already running, with API server enabled."""
    s = socket.socket()
    try:
        s.settimeout(2)
        s.connect(("localhost", 8642))
        s.close()
        print("Hermes gateway: already running")
        return
    except:
        pass

    print("Starting Hermes gateway...")
    import subprocess as _sp
    env = os.environ.copy()
    env["API_SERVER_KEY"] = API_SERVER_KEY
    _sp.Popen(
        ["hermes", "gateway", "run", "--replace"],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        env=env,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect(("localhost", 8642))
            s.close()
            print("Hermes gateway: ready")
            return
        except:
            continue
    print("! Failed to start Hermes, please start it manually")


# ── 1. Wake Word Detection ──


# ── 1. Wake Word Detection ──

def wait_wakeword(timeout=120):
    """Listen until 'hey jarvis' detected"""
    import openwakeword
    from openwakeword.model import Model
    import sounddevice as sd
    import numpy as np

    # Auto-download models if missing
    mdir = Path(openwakeword.__file__).parent / "resources" / "models"

    # Check for at least one valid model file
    has_models = any(
        mdir.glob("*.onnx")
    ) or any(mdir.glob("*.tflite"))
    if not has_models:
        print("Downloading openWakeWord models...")
        openwakeword.utils.download_models()

    onx = mdir / "embedding_model.onnx"
    if not onx.exists():
        tfl = mdir / "embedding_model.tflite"
        if tfl.exists():
            import shutil
            shutil.copy(str(tfl), str(onx))

    print("Loading openWakeWord...")
    model = Model(wakeword_models=["alexa"])
    print('Listening for "alexa"... (Ctrl+C to skip)')

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


# ── 1b. Play prompt tone ──

PROMPT_PATH = Path(__file__).parent / "zaizai.wav"

def play_prompt():
    """Play '在呢' prompt tone"""
    if not PROMPT_PATH.exists():
        print("  (no prompt tone)")
        return
    import subprocess as _sp
    _sp.run(["afplay", str(PROMPT_PATH)], capture_output=True)


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


# ── 3. ASR (Volcengine V3 — 大模型语音识别) ──

ASR_RESOURCE_ID = os.environ.get("ASR_RESOURCE_ID", "volc.seedasr.sauc.duration")

def asr(pcm):
    """Volcengine ASR V3 (bigmodel_nostream) — unified X-Api-Key auth."""
    import websocket
    import io as _io

    # Wrap raw PCM into WAV (V3 requires WAV format)
    buf = _io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    wav_data = buf.getvalue()

    def _hdr(msg_type, flags, serialization=0):
        return bytes([
            0x11,
            (msg_type << 4) | flags,
            (serialization << 4) | 0x00,
            0x00,
        ])

    ws = websocket.WebSocket()
    ws.settimeout(15)
    ws.connect(
        "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream",
        header={
            "X-Api-Key": VOLC_TOKEN,
            "X-Api-Resource-Id": ASR_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        },
    )

    # Send config (JSON, serialization=1)
    config = json.dumps({
        "user": {"uid": "clawd-test"},
        "audio": {"format": "wav", "rate": 16000, "bits": 16,
                  "channel": 1, "language": "zh-CN"},
        "request": {"model_name": "bigmodel"},
    }).encode("utf-8")
    ws.send(_hdr(0x01, 0x00, serialization=1) +
            struct.pack(">I", len(config)) + config,
            opcode=websocket.ABNF.OPCODE_BINARY)

    # Send audio (last frame, flags=0x02)
    ws.send(_hdr(0x02, 0x02) +
            struct.pack(">I", len(wav_data)) + wav_data,
            opcode=websocket.ABNF.OPCODE_BINARY)

    text = ""
    while True:
        msg = ws.recv()
        if not isinstance(msg, bytes) or len(msg) < 12:
            continue
        msg_type = (msg[1] >> 4) & 0x0F
        flags = msg[1] & 0x0F
        psz = struct.unpack(">I", msg[8:12])[0]
        payload = msg[12:12 + psz] if psz <= len(msg) - 12 else msg[12:]

        if msg_type == 0x09:  # full server response
            r = json.loads(payload.decode("utf-8"))
            t = r.get("result", {}).get("text", "")
            if t:
                text = t
            if flags == 0x03:  # last response
                break
        elif msg_type == 0x0F:  # error
            print(f"ASR error: {payload.decode('utf-8', errors='replace')[:200]}")
            break

    ws.close()

    if text:
        print(f'ASR: "{text}"')
    else:
        print("ASR: no result")
    return text


# ── 4. LLM (Hermes /v1/runs — with memory) ──

def chat(text):
    import requests

    print("LLM...")
    hdrs = {"Content-Type": "application/json"}
    if HERMES_KEY:
        hdrs["Authorization"] = f"Bearer {HERMES_KEY}"
    else:
        hdrs["X-Api-Key"] = API_SERVER_KEY

    # Create run
    base = HERMES_URL.rsplit("/v1/chat/completions", 1)[0]
    resp = requests.post(
        f"{base}/v1/runs",
        json={"input": text, "instructions": LLM_PROMPT},
        headers=hdrs, timeout=10,
    )
    if resp.status_code not in (200, 202):
        print(f"  Run HTTP {resp.status_code}")
        return None

    run_id = resp.json().get("run_id") or resp.json().get("id", "")
    if not run_id:
        print(f"  No run_id: {resp.text[:200]}")
        return None

    # Stream events
    ev_url = f"{base}/v1/runs/{run_id}/events"
    ev_resp = requests.get(ev_url, headers=hdrs, timeout=120, stream=True)
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
            # Auto-approve for voice assistant (no UI)
            approve_url = f"{base}/v1/runs/{run_id}/approval"
            try:
                requests.post(approve_url, json={"choice": "always"},
                              headers=hdrs, timeout=5)
            except Exception:
                pass
            continue

        elif event == "message.delta":
            delta = ev.get("delta", "")
            if delta:
                reply += delta
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
        print(f"LLM: {reply[:100]}")
    else:
        print("LLM: empty reply")
    return reply


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

    import subprocess, tempfile, os as _os

    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(bytes(pcm))
    print(f"  TTS: {len(pcm)} bytes")

    subprocess.run(["afplay", tmp_path])
    _os.unlink(tmp_path)
    print("  Played")
    return True


# ── Main ──

def main():
    print("=" * 50)
    print("  Clawd Pipeline Test")
    print("=" * 50)

    ensure_hermes()

    if not wait_wakeword():
        sys.exit(1)

    play_prompt()

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
