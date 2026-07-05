"""
Clawd — 跨平台语音助手核心模块
ASR / LLM / TTS 共享逻辑
"""
import os, sys, json, time, base64, struct, uuid, wave, socket
from pathlib import Path

# ── 配置 ──

def load_env(env_dir=None):
    """加载 .env，返回配置字典"""
    if env_dir:
        p = Path(env_dir) / ".env"
    else:
        p = Path.cwd() / ".env"
    if p.exists():
        for line in p.read_text().strip().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    cfg = {
        "volc_token": os.environ.get("VOLC_TOKEN", ""),
        "hermes_key": os.environ.get("HERMES_API_KEY", ""),
        "hermes_url": os.environ.get("HERMES_URL", "http://localhost:8642"),
        "api_server_key": os.environ.get("API_SERVER_KEY", "clawd"),
        "asr_resource_id": os.environ.get("ASR_RESOURCE_ID", "volc.seedasr.sauc.duration"),
        "llm_prompt": os.environ.get("LLM_PROMPT", "你是一个中文语音助手，回答简短直接。"),
        "record_secs": int(os.environ.get("RECORD_SECS", "5")),
    }
    if not cfg["volc_token"]:
        print("! Missing VOLC_TOKEN, check .env")
        sys.exit(1)
    return cfg


# ── Hermes 自启 ──

def ensure_hermes(api_server_key="clawd"):
    """启动 Hermes gateway（如未运行）"""
    s = socket.socket()
    try:
        s.settimeout(2)
        s.connect(("localhost", 8642))
        s.close()
        print("Hermes: ready")
        return
    except:
        pass
    print("Starting Hermes...")
    import subprocess as _sp
    env = os.environ.copy()
    env["API_SERVER_KEY"] = api_server_key
    _sp.Popen(
        ["hermes", "gateway", "run", "--replace"],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, env=env,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect(("localhost", 8642))
            s.close()
            print("Hermes: ready")
            return
        except:
            continue
    print("! Failed to start Hermes")


# ── 唤醒词模型初始化 ──

def init_wakeword_model(wakeword="alexa"):
    """初始化 openWakeWord 模型，按需下载。返回 Model 实例"""
    import openwakeword
    from openwakeword.model import Model

    mdir = Path(openwakeword.__file__).parent / "resources" / "models"
    has_models = any(mdir.glob("*.onnx")) or any(mdir.glob("*.tflite"))
    if not has_models:
        print("Downloading models...")
        openwakeword.utils.download_models()
    onx = mdir / "embedding_model.onnx"
    if not onx.exists():
        tfl = mdir / "embedding_model.tflite"
        if tfl.exists():
            import shutil
            shutil.copy(str(tfl), str(onx))

    print("Loading wake word...")
    model = Model(wakeword_models=[wakeword])
    print(f'🎤 Listening "{wakeword}"...')
    return model


# ── ASR (V3 bigmodel_nostream) ──

def asr_transcribe(pcm, volc_token="", resource_id="volc.seedasr.sauc.duration"):
    """PCM 音频 → 文字 (火山引擎 V3)"""
    import websocket as _ws
    import io as _io

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    wav_data = buf.getvalue()

    def _hdr(mt, fl, sr=0):
        return bytes([0x11, (mt << 4) | fl, (sr << 4) | 0x00, 0x00])

    ws = _ws.WebSocket()
    ws.settimeout(15)
    ws.connect(
        "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream",
        header={
            "X-Api-Key": volc_token,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        },
    )

    config = json.dumps({
        "user": {"uid": "clawd"},
        "audio": {"format": "wav", "rate": 16000, "bits": 16,
                  "channel": 1, "language": "zh-CN"},
        "request": {"model_name": "bigmodel"},
    }).encode("utf-8")
    ws.send(_hdr(0x01, 0x00, 1) + struct.pack(">I", len(config)) + config,
            opcode=_ws.ABNF.OPCODE_BINARY)
    ws.send(_hdr(0x02, 0x02) + struct.pack(">I", len(wav_data)) + wav_data,
            opcode=_ws.ABNF.OPCODE_BINARY)

    text = ""
    while True:
        msg = ws.recv()
        if not isinstance(msg, bytes) or len(msg) < 12:
            continue
        mt = (msg[1] >> 4) & 0x0F
        fl = msg[1] & 0x0F
        psz = struct.unpack(">I", msg[8:12])[0]
        payload = msg[12:12 + psz] if psz <= len(msg) - 12 else msg[12:]

        if mt == 0x09:
            r = json.loads(payload.decode("utf-8"))
            t = r.get("result", {}).get("text", "")
            if t:
                text = t
            if fl == 0x03:
                break
        elif mt == 0x0F:
            print(f"ASR error: {payload.decode('utf-8', errors='replace')[:200]}")
            break
    ws.close()

    if text:
        print(f'📝 "{text}"')
    else:
        print("ASR: no result")
    return text


# ── LLM (Hermes /v1/runs) ──

def llm_chat(text, hermes_url="", hermes_key="", api_server_key="clawd",
             prompt="你是一个中文语音助手，回答简短直接。"):
    """文字 → 回复 (Hermes /v1/runs，带记忆)"""
    import requests

    print("🤖 LLM...")
    hdrs = {"Content-Type": "application/json"}
    if hermes_key:
        hdrs["Authorization"] = f"Bearer {hermes_key}"
    elif api_server_key:
        hdrs["Authorization"] = f"Bearer {api_server_key}"

    base = hermes_url.rstrip("/")

    resp = requests.post(
        f"{base}/v1/runs",
        json={"input": text, "instructions": prompt},
        headers=hdrs, timeout=10,
    )
    if resp.status_code not in (200, 202):
        print(f"  Run HTTP {resp.status_code}")
        return None
    run_id = resp.json().get("run_id") or resp.json().get("id", "")
    if not run_id:
        print(f"  No run_id: {resp.text[:200]}")
        return None

    ev_resp = requests.get(f"{base}/v1/runs/{run_id}/events",
                           headers=hdrs, timeout=120, stream=True)
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
            try:
                requests.post(f"{base}/v1/runs/{run_id}/approval",
                              json={"choice": "always"},
                              headers=hdrs, timeout=5)
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
        print("LLM: empty")
    return reply


# ── TTS (V3 HTTP Chunked) ──

def tts_synthesize(text, volc_token=""):
    """文字 → PCM 音频 (火山引擎 V3 TTS)"""
    import requests

    hdrs = {
        "Content-Type": "application/json",
        "x-api-key": volc_token,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Request-Id": str(uuid.uuid4()),
    }
    payload = {
        "user": {"uid": "clawd"},
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
        return None
    pcm = bytearray()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get("data"):
                pcm.extend(base64.b64decode(c["data"]))
        except json.JSONDecodeError:
            continue
    if not pcm:
        print("TTS: no audio")
        return None
    print(f"🔊 TTS: {len(pcm)} bytes")
    return bytes(pcm)
