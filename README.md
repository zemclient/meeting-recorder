# Meeting Recorder — запись встреч, задачи и сроки

Инструмент для деловых встреч (в т.ч. Yandex Телемост):

1. **Записывает** всю встречу в аудио (системный звук + микрофон).
2. **Расшифровывает** речь в текст (локально через **faster-whisper**, либо **OpenAI Whisper**, либо **Yandex SpeechKit**).
3. **Извлекает** через LLM задачи, ответственных, сроки и краткий итог встречи:
   - **Локально (llama.cpp, GGUF)** — полностью оффлайн, без сторонних серверов (по умолчанию).
   - **OpenAI** — облачный анализ (нужен API-key).
4. Показывает результат в веб-интерфейсе и отдаёт **отчёт в Word (.docx)** для скачивания (в имени файла — дата документа, напр. `meeting_report_18-08-2026.docx`).

> Все ключевые компоненты могут работать локально и оффлайн: STT (faster-whisper) + анализ (llama.cpp). Yandex не требуется.

## Как запустить (через Pinokio)

1. Откройте папку проекта в Pinokio.
2. Нажмите **Установить** (install.js) — ставятся зависимости в `app/env`.
3. Нажмите **Запустить** (start.js) — поднимется локальный сервер.
4. Откройте **Открыть интерфейс** — откроется веб-страница `http://127.0.0.1:<port>`.

## Как запустить вручную

```bash
cd app
python -m venv env
env\Scripts\activate      # Windows  (или: source env/bin/activate на Linux/macOS)
pip install -r requirements.txt
PORT=5000 python server.py
```

Откройте http://127.0.0.1:5000

## Настройка

В веб-интерфейсе укажите:

- **STT (распознавание речи):** выберите провайдера в поле «STT провайдер»:
  - *Локально (offline, faster-whisper)* — без ключей; веса модели (по умолчанию `small`) скачиваются автоматически при первом запуске. Работает на CPU/GPU локально.
  - *OpenAI Whisper (без Яндекса)* — нужен только `OpenAI API-key` (поле «OpenAI: API-key», общий с анализом). Модель по умолчанию `whisper-1`. Речь отправляется в `https://api.openai.com/v1/audio/transcriptions`.
  - *Yandex SpeechKit* — `API-key` и `Folder ID` из [Yandex Cloud](https://console.cloud.yandex.ru/). Сервис «SpeechKit» должен быть активирован в каталоге.
- **Анализ (LLM):** по умолчанию **Локальная LLM (offline, llama.cpp)** — без ключей и сторонних серверов. Модель GGUF (по умолчанию `qwen2.5-1.5b-instruct-q4_k_m.gguf`) скачивается автоматически при первом анализе (~1.1 ГБ). Можно указать другую GGUF-модель в поле «Анализ: локальная модель».
  - *OpenAI:* `API-key` и модель (по умолчанию `gpt-4o-mini`) для облачного анализа.
- **Режим записи:**
  - `Авто` — системный звук, если доступен, иначе микрофон; при наличии обоих пишет и микширует (система + микрофон).
  - `Только системный звук` — loopback (что слышно в Телемосте).
  - `Только микрофон`.
  - `Система + микрофон (микс)`.

> **Windows:** захват системного звука использует WASAPI loopback. Если запись системного звука пустая,
> включите в «Звук → Запись» устройство **Stereo Mix** (или установите виртуальный кабель, напр. VB-Audio),
> либо выберите режим «Только микрофон».

## Использование

1. Откройте встречу в Yandex Телемост.
2. В интерфейсе нажмите **● Начать запись**.
3. По окончании нажмите **■ Остановить и обработать** — пойдёт распознавание и анализ.
4. Получите: итог, список задач (с ответственным и сроком), сроки, решения и полную расшифровку.
5. Скачайте отчёт кнопкой **⬇ Word (.docx)**.

## Программный доступ к API

Сервер предоставляет HTTP API (базовый URL `http://127.0.0.1:<port>`).

### JavaScript (fetch)

```js
// Сохранить настройки
await fetch("/api/config", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    stt_provider: "local",               // local | openai | yandex
    analysis_provider: "local",          // local (llama.cpp) | openai
    record_mode: "auto"
    // openai_api_key: "sk-..."            // нужен только для openai-провайдеров
  })
});

// Начать запись
await fetch("/api/start", { method: "POST" });

// Остановить и получить результат
const res = await fetch("/api/stop", { method: "POST" }).then(r => r.json());
console.log(res.report);   // { summary, tasks[], deadlines[], decisions[] }
console.log(res.transcript);

// Скачать отчёт (DOCX, имя файла содержит дату документа)
window.open("/api/report/docx");
```

### Python

```python
import requests, json

BASE = "http://127.0.0.1:5000"
requests.post(f"{BASE}/api/config", json={
    "stt_provider": "local",
    "analysis_provider": "local",
    # "openai_api_key": "sk-..."         # для openai-провайдеров
}).raise_for_status()

requests.post(f"{BASE}/api/start").raise_for_status()
# ... встреча идёт ...
r = requests.post(f"{BASE}/api/stop").json()
print(r["report"])
open("meeting_report_18-08-2026.docx", "wb").write(requests.get(f"{BASE}/api/report/docx").content)
```

### Curl

```bash
curl -X POST http://127.0.0.1:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"stt_provider":"local","analysis_provider":"local"}'

curl -X POST http://127.0.0.1:5000/api/start
curl -X POST http://127.0.0.1:5000/api/stop
curl -O http://127.0.0.1:5000/api/report/docx
```

## Структура

```
app/
  recorder.py     # захват аудио (soundcard: система + микрофон)
  transcribe.py   # STT: faster-whisper (local) / OpenAI Whisper / Yandex SpeechKit
  analyze.py      # извлечение задач/сроков/итога: llama.cpp (local) / OpenAI
  server.py       # Flask-сервер + API
  templates/index.html  # веб-интерфейс
  requirements.txt
install.js / start.js / pinokio.js / pinokio.json   # Pinokio-ланчер
```
