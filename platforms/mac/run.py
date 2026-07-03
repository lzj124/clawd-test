#!/usr/bin/env python3
"""
Clawd — Mac 版持续语音助手
唤醒词 → 录音 → ASR → LLM → TTS → 循环
"""
import os, sys, json, time, base64, struct, uuid, wave, tempfile
from pathlib import Path

# 添加项目根目录到路径
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from clawd import (
    load_env, ensure_hermes, asr_transcribe, llm_chat, tts_synthesize
)

cfg = load_env(str(_root))


# ── 唤醒词 (Mac: sounddevice) ──

def wait_wakeword(timeout=120):
    import openwakeword
    from openwakeword.model import Model
    import sounddevice as sd
    import numpy as np

    mdir = Path(openwakeword.__file__).parent / "resources" / "models"
    has_models = any(mdir.glob("*.onnx")) or any(mdir.glob("*.tflite"))
    if not has_models:
        print("Downloading models...")
        openwakeword.utils.download_models()
    onx = mdir / "embedding_model.onnx"
    if not onx.exists():
        tfl = mdir / "embedding_model.tflite"
        if tfl.exists():
            import shutil; shutil.copy(str(tfl), str(onx))

    print("Loading wake word...")
    model = Model(wakeword_models=["alexa"])
    print('🎤 Listening "alexa"...')
    SR, FRAME = 16000, 1280
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME)
    stream.start()
    start = time.time()
    try:
        while time.time() - start < timeout:
            audio, _ = stream.read(FRAME)
            for ww, score in model.predict(audio.flatten()).items():
                if score > 0.5:
                    print(f'🔔 "{ww}" ({score:.2f})')
                    return True
            if int(time.time() - start) > 0 and int(time.time() - start) % 10 == 0:
                print(f"  ... ({int(time.time()-start)}s)")
    except KeyboardInterrupt:
        return False
    finally:
        stream.stop(); stream.close()
    return False


# ── 提示音 (Mac: afplay) ──

PROMPT_PATH = Path(__file__).parent / "zaizai.wav"

def play_prompt():
    import subprocess as sp
    if PROMPT_PATH.exists():
        sp.run(["afplay", str(PROMPT_PATH)], capture_output=True)


# ── 录音 (Mac: sounddevice) ──

def record(secs=5):
    import sounddevice as sd
    import numpy as np
    print(f"🎙️ Recording {secs}s...")
    audio = sd.rec(int(16000 * secs), samplerate=16000, channels=1, dtype="int16")
    for i in range(secs):
        time.sleep(1)
        print(f"  {i+1}/{secs}")
    sd.wait()
    print("  Done")
    return audio.flatten().tobytes()


# ── TTS 播放 (Mac: afplay + 临时文件) ──

def speak(text):
    pcm = tts_synthesize(text, cfg["volc_token"])
    if not pcm:
        return False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(bytes(pcm))
    import subprocess as sp, os as _os
    sp.run(["afplay", path])
    _os.unlink(path)
    print("  Played")
    return True


# ── Main ──

def main():
    print("=" * 50)
    print("  Clawd Mac — 持续语音助手")
    print("  唤醒词 → 录音 → ASR → LLM → TTS → 循环")
    print("  Ctrl+C 退出")
    print("=" * 50)

    ensure_hermes(cfg["api_server_key"])
    print()

    while True:
        if not wait_wakeword():
            break
        play_prompt()
        audio = record(cfg["record_secs"])
        text = asr_transcribe(audio, cfg["volc_token"], cfg["asr_resource_id"])
        if not text:
            print("  (no speech, back to listening)\n")
            continue
        reply = llm_chat(text, cfg["hermes_url"], cfg["hermes_key"],
                         cfg["api_server_key"], cfg["llm_prompt"])
        if not reply:
            print("  (LLM failed, back to listening)\n")
            continue
        speak(reply)
        print(f"\nOK: '{text}' -> '{reply}'\n")
        print("─" * 40)

if __name__ == "__main__":
    main()
