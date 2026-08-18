import io
import wave
import math
import requests
import numpy as np

STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
# Максимальный размер тела запроса SpeechKit (синхронное API) ~1 МБ,
# максимальная длительность ~60 c. Берём с запасом 30 c.
CHUNK_SECONDS = 30


class TranscribeError(Exception):
    pass


def _read_mono_16k(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)

    samples = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch > 2:
        samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)

    if sr != 16000:
        # Простое линейное ресемплирование до 16 кГц
        new_len = int(len(samples) * 16000 / sr)
        idx = np.linspace(0, len(samples) - 1, new_len).astype(np.int32)
        samples = samples[idx]
        sr = 16000
    return samples, sr


def _wav_chunk_bytes(samples, sr, start, end):
    chunk = samples[start:end]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(chunk.tobytes())
    return buf.getvalue()


def transcribe_yandex(wav_path, api_key, folder_id, lang="ru-RU", progress_cb=None):
    """Распознавание речи через Yandex SpeechKit (синхронное API, по кускам)."""
    if not api_key or not folder_id:
        raise TranscribeError("Не указаны api_key / folder_id для SpeechKit.")

    samples, sr = _read_mono_16k(wav_path)
    total = len(samples)
    chunk_samples = int(CHUNK_SECONDS * sr)

    parts = []
    done = 0
    for i in range(0, total, chunk_samples):
        end = min(i + chunk_samples, total)
        data = _wav_chunk_bytes(samples, sr, i, end)
        resp = requests.post(
            STT_URL,
            params={"folderId": folder_id, "lang": lang},
            headers={"Authorization": f"Api-Key {api_key}"},
            data=data,
            timeout=120,
        )
        if resp.status_code != 200:
            raise TranscribeError(
                f"SpeechKit вернул {resp.status_code}: {resp.text}"
            )
        try:
            text = resp.json().get("result", "")
        except Exception:
            text = ""
        if text:
            parts.append(text)
        done = end
        if progress_cb:
            progress_cb(done / total if total else 1, text)

    return " ".join(p.strip() for p in parts if p.strip())


def transcribe_openai(wav_path, api_key, model="whisper-1", lang="ru", progress_cb=None):
    """Распознавание речи через OpenAI Whisper (audio/transcriptions). Без Яндекса."""
    if not api_key:
        raise TranscribeError("Не указан openai_api_key для Whisper.")

    with open(wav_path, "rb") as f:
        resp = requests.post(
            OPENAI_STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model, "language": lang},
            files={"file": ("meeting.wav", f, "audio/wav")},
            timeout=300,
        )
    if resp.status_code != 200:
        raise TranscribeError(f"Whisper вернул {resp.status_code}: {resp.text}")
    text = resp.json().get("text", "")
    if progress_cb:
        progress_cb(1, text)
    return text.strip()


def _wav_duration(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
    return (n / sr) if sr else 0.0


def transcribe_local(wav_path, model_size="small", lang="ru", progress_cb=None):
    """Полностью локальное распознавание через faster-whisper. Без сторонних серверов.

    Модель (веса) скачивается автоматически при первом запуске и кэшируется.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscribeError(
            "Не установлен faster-whisper. Выполните: uv pip install faster-whisper"
        )
    duration = _wav_duration(wav_path)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        wav_path,
        language=lang,
        beam_size=10,
        best_of=10,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    parts = []
    for seg in segments:
        if seg.text:
            parts.append(seg.text.strip())
        if progress_cb and duration:
            progress_cb(min(1.0, seg.end / duration), seg.text)
    return " ".join(parts)


def transcribe(wav_path, provider="yandex", progress_cb=None, **kwargs):
    """Диспетчер провайдеров STT.

    provider == "openai" -> OpenAI Whisper (нужен openai_key, опц. openai_model)
    provider == "local"  -> faster-whisper оффлайн (опц. local_model)
    иначе                -> Yandex SpeechKit (нужны api_key, folder_id)
    """
    if provider == "openai":
        return transcribe_openai(
            wav_path,
            kwargs.get("openai_key"),
            kwargs.get("openai_model", "whisper-1"),
            lang=kwargs.get("lang", "ru"),
            progress_cb=progress_cb,
        )
    if provider == "local":
        return transcribe_local(
            wav_path,
            kwargs.get("local_model", "small"),
            lang=kwargs.get("lang", "ru"),
            progress_cb=progress_cb,
        )
    return transcribe_yandex(
        wav_path,
        kwargs.get("api_key"),
        kwargs.get("folder_id"),
        lang=kwargs.get("lang", "ru-RU"),
        progress_cb=progress_cb,
    )
