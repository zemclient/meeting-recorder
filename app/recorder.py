import os
import time
import wave
import threading
import numpy as np
import sounddevice as sd

try:
    sd.check_input_settings
except Exception:  # pragma: no cover
    pass


class RecorderError(Exception):
    pass


class Recorder:
    """Запись аудио встречи через sounddevice (PortAudio).

    Режимы:
      auto     - системный звук (Stereo Mix / loopback) + микрофон с микшированием.
                 Если loopback недоступен, пишется только микрофон.
      loopback - только системный звук
      mic      - только микрофон
      both     - система + микрофон (микс)
    """

    SR = 16000

    def __init__(self, mode="auto", samplerate=16000):
        self.mode = mode
        self.SR = samplerate
        self._running = False
        self._lock = threading.Lock()
        # Упорядоченные списки 1-секундных кусков (float32) на каждый источник.
        self._loop_chunks = []
        self._mic_chunks = []
        self._streams = []
        self._start_ts = None

    # ----- выбор устройств -----
    def _pick_devices(self):
        inputs = []
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    inputs.append((i, d["name"]))
        except Exception:
            pass

        loop = None
        mic = None
        for i, name in inputs:
            n = name.lower()
            if loop is None and ("stereo mix" in n or "loopback" in n):
                loop = i
            if mic is None and "microphone" in n:
                mic = i
        if mic is None:
            try:
                mic = sd.default.device[0]
            except Exception:
                mic = None
        return loop, mic

    def _make_callback(self, target):
        def cb(indata, frames, time_info, status):
            if self._running and indata is not None and len(indata):
                chunk = np.asarray(indata, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                with self._lock:
                    target.append(chunk.astype(np.float32))
        return cb

    # ----- управление -----
    def start(self):
        if self._running:
            return
        loop, mic = self._pick_devices()

        use_loop = self.mode in ("auto", "loopback", "both") and loop is not None
        use_mic = self.mode in ("auto", "mic", "both") and mic is not None

        if not use_loop and not use_mic:
            raise RecorderError(
                "Не найдены устройства записи (микрофон / Stereo Mix)."
            )

        self._running = True
        self._loop_chunks = []
        self._mic_chunks = []
        self._streams = []
        self._start_ts = time.time()

        if use_loop:
            try:
                s = sd.InputStream(
                    device=loop,
                    channels=1,
                    samplerate=self.SR,
                    blocksize=self.SR,
                    dtype="float32",
                    callback=self._make_callback(self._loop_chunks),
                )
                s.start()
                self._streams.append(s)
            except Exception as e:
                print(f"[recorder] не удалось запустить loopback: {e}")

        if use_mic:
            try:
                s = sd.InputStream(
                    device=mic,
                    channels=1,
                    samplerate=self.SR,
                    blocksize=self.SR,
                    dtype="float32",
                    callback=self._make_callback(self._mic_chunks),
                )
                s.start()
                self._streams.append(s)
            except Exception as e:
                print(f"[recorder] не удалось запустить микрофон: {e}")

        if not self._streams:
            self._running = False
            raise RecorderError("Не удалось запустить ни одного потока записи.")

    def stop(self):
        if not self._running:
            return None
        self._running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []
        return self._build_wav()

    def is_running(self):
        return self._running

    def elapsed(self):
        if self._start_ts is None:
            return 0
        return int(time.time() - self._start_ts)

    # ----- сборка wav -----
    def _build_wav(self):
        with self._lock:
            loop = list(self._loop_chunks)
            mic = list(self._mic_chunks)
        self._loop_chunks = []
        self._mic_chunks = []

        if not loop and not mic:
            return None

        n = max(len(loop), len(mic))
        mixed = []
        for i in range(n):
            parts = []
            if i < len(loop):
                parts.append(loop[i])
            if i < len(mic):
                parts.append(mic[i])
            if len(parts) > 1:
                min_len = min(len(p) for p in parts)
                sig = np.zeros(min_len, dtype=np.float32)
                for p in parts:
                    sig += p[:min_len]
                sig /= len(parts)
            else:
                sig = parts[0]
            mixed.append(sig)

        if not mixed:
            return None

        audio = np.concatenate(mixed)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "meeting.wav"
        )
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SR)
            wf.writeframes(pcm.tobytes())
        return out_path
