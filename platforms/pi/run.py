#!/usr/bin/env python3
"""
Clawd — Pi 版持续语音助手
唤醒词 → 录音 → ASR → LLM → TTS → 循环
"""
import os, sys, json, time, base64, struct, uuid, wave, subprocess
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from clawd import (
    load_env, ensure_hermes, asr_transcribe, llm_chat, tts_synthesize
)

cfg = load_env(str(_root))

AUDIO_DEV = os.environ.get("AUDIO_DEV", "plughw:2,0")


# ── 音频工具 ──

def play_wav(path):
    subprocess.run(["aplay", "-D", AUDIO_DEV, str(path)], capture_output=True)

def record_wav(path, secs=5):
    print(f"🎙️ Recording {secs}s...")
    subprocess.run(["arecord", "-D", AUDIO_DEV, "-d", str(secs),
                    "-r", "16000", "-f", "S16_LE", "-c", "1", str(path)],
                   capture_output=True)
    print("  Done")

def wav_to_pcm(path):
    with open(path, "rb") as f:
        data = f.read()
    return data[44:] if data[:4] == b'RIFF' else data


# ── 唤醒词 (Pi: arecord pipe) ──

def wait_wakeword():
    import openwakeword
    from openwakeword.model import Model
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

    proc = subprocess.Popen(
        ["arecord", "-D", AUDIO_DEV, "-r", "16000", "-f", "S16_LE", "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            raw = proc.stdout.read(2560)
            if len(raw) < 2560:
                break
            audio = np.frombuffer(raw, dtype=np.int16)
            for ww, score in model.predict(audio).items():
                if score > 0.5:
                    print(f'🔔 "{ww}" ({score:.2f})')
                    return True
    except KeyboardInterrupt:
        return False
    finally:
        proc.terminate(); proc.wait()
    return False


# ── 提示音 ──

def ensure_prompt():
    p = Path("/tmp/clawd_prompt.wav")
    if p.exists():
        return p
    pcm = tts_synthesize("在呢", cfg["volc_token"])
    if pcm:
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(bytes(pcm))
        return p
    return None


# ── TTS 播放 ──

def speak(text):
    pcm = tts_synthesize(text, cfg["volc_token"])
    if not pcm:
        return False
    out = Path("/tmp/clawd_reply.wav")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(bytes(pcm))
    play_wav(out)
    print("  Played")
    return True


# ── Main Loop ──

def main():
    print("=" * 50)
    print("  Clawd Pi — 持续语音助手")
    print("  唤醒词 → 录音 → ASR → LLM → TTS → 循环")
    print("  Ctrl+C 退出")
    print("=" * 50)

    ensure_hermes(cfg["api_server_key"])
    prompt = ensure_prompt()

    while True:
        if not wait_wakeword():
            break
        if prompt:
            print("📢 在呢!")
            play_wav(prompt)
        wav = Path("/tmp/clawd_input.wav")
        record_wav(wav, cfg["record_secs"])
        pcm = wav_to_pcm(wav)
        text = asr_transcribe(pcm, cfg["volc_token"], cfg["asr_resource_id"])
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
    print("再见！")

if __name__ == "__main__":
    main()
