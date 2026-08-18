import json
import os
import re
import requests
from datetime import datetime, date

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Локальная LLM (llama.cpp GGUF). Модель скачивается автоматически при первом использовании.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LLM_MODEL = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_LLM_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
LLM_CTX = 4096

SYSTEM_PROMPT = (
    "Ты - ассистент по протоколированию деловых встреч на русском языке. "
    "Тебе передаётся расшифровка (транскрипция) встречи. "
    "Твоя задача - максимально полно и точно извлечь СТРУКТУРИРОВАННУЮ информацию.\n\n"
    "Верни ТОЛЬКО валидный JSON без какого-либо другого текста и без markdown, в формате:\n"
    "{\n"
    '  "summary": "краткий итог встречи в 3-6 предложениях",\n'
    '  "tasks": [\n'
    '    {"description": "что нужно сделать (суть задачи)",\n'
    '     "owner": "ответственный (ФИО или должность) или null, если не назван",\n'
    '     "deadline": "крайний срок в точной формулировке из речи, напр. \'15 октября 2026 года\', или null"}\n'
    "  ],\n"
    '  "deadlines": [\n'
    '    {"what": "по какой задаче/теме срок (кратко)", "when": "срок в точной формулировке из речи"}\n'
    "  ],\n"
    '  "decisions": ["принятые решения"]\n'
    "}\n\n"
    "ПРАВИЛА (строго соблюдай):\n"
    "1. Каждая упомянутая задача с ответственным или сроком должна попасть в tasks. "
    "Даже если ответственный или срок не названы - всё равно добавь задачу с null в соответствующем поле.\n"
    "2. ПОЛЕ 'owner' (Ответственный) заполняй, если в речи назван исполнитель под ЛЮБЫМ из этих "
    "синонимов: ответственный, исполнитель, куратор, 'кто отвечает', 'за кем закреплено', "
    "'кому поручено', 'ведёт', 'контролирует', 'будет делать'. Извлекай само имя/ФИО/должность "
    "лица БЕЗ слов-ролей: например, при 'куратор проекта Анна Смирнова' укажи 'Анна Смирнова', "
    "а не 'куратор проекта Анна'. Если назван только исполнитель без имени - пиши его должность.\n"
    "3. Любое упоминание ДАТЫ или СРОКА (число месяца, 'до пятницы', 'через две недели', "
    "'15 октября 2026 года' и т.п.) обязательно заноси в deadlines, а если оно относится к задаче - "
    "и в task.deadline. НЕ опускай даты.\n"
    "4. 'when' и 'deadline' пиши максимально близко к тому, как сказано в речи (с числом и месяцем/годом).\n"
    "5. 'what' в deadlines - короткая суть, к чему относится срок.\n"
    "6. ПОЛЕ 'summary' (Итог) формируй из резюме/выводов встречи под ЛЮБЫМ из синонимов: "
    "итог, резюме, выводы, суть, кратко, основное, главное. Если в речи есть явная фраза-резюме "
    "(напр. 'в итоге...', 'резюмируя...', 'подведём итог...') - используй её.\n"
    "7. Если задач/сроков/решений нет - верни пустые списки. "
    "НЕ выдумывай ответственных и сроки, которых нет в тексте.\n"
    "8. Разделы могут называться синонимами - распознавай их и клади в соответствующие поля: "
    "'summary' = Результат/Итог/Резюме/Выводы; 'tasks' = Задачи/Поручения/План/To-Do; "
    "'deadlines' = Сроки/Дедлайны/Даты; 'decisions' = Принятые решения/Договорённости/Постановления; "
    "контекст самой встречи = Встреча/Совещание/Созвон."
)

# --- Синонимы ключевых полей (для устойчивого распознавания) ---
# Синонимы слова "Ответственный": исполнитель, куратор, отвечает, закреплено и т.п.
RESPONSIBLE_SYNONYMS = [
    "ответственн", "исполнител", "куратор", "отвеча", "закреплен",
    "назначен", "поручен", "куриру", "вед[её]т", "контролир",
    "будет делать", "делает", "за ним", "за ней",
]
# Синонимы слова "Итог" (summary): резюме, выводы, суть, кратко и т.п.
SUMMARY_SYNONYMS = [
    "итог", "резюм", "вывод", "суть", "кратк", "подвед", "обобщ",
    "суммир", "основн", "главн",
]
_OWNER_RE = re.compile(
    r"(?i:(?:"
    + "|".join(RESPONSIBLE_SYNONYMS)
    + r"))[^А-ЯЁ]{0,40}?([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})"
    r"|([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})\s+(?i:(?:"
    + "|".join(RESPONSIBLE_SYNONYMS)
    + r"))"
)
_SUMMARY_SENT_RE = re.compile(
    r"(?:^|[\n.;])(?=[^\n]*?(?:"
    + "|".join(SUMMARY_SYNONYMS)
    + r"))([^\n.!?]+[.!?]?)",
    re.IGNORECASE,
)


def _clean_name(name):
    """Очищает извлечённое имя: убирает неименные слова в начале
    (напр. 'проекта Анна Смирнова' -> 'Анна Смирнова') и лишние пробелы."""
    if not name:
        return name
    name = " ".join(name.split())
    words = name.split()
    # отбрасываем ведущие слова, не начинающиеся с заглавной буквы
    while words and not (words[0][:1].isupper()):
        words = words[1:]
    # оставляем максимум 3 слова (ФИО / имя-фамилия)
    words = words[:3]
    return " ".join(words).strip()


def _extract_owners_from_text(transcript):
    """Fallback: находит в речи упоминания ответственных (по синонимам)
    и возвращает список кандидатов-имён."""
    found = []
    seen = set()
    for m in _OWNER_RE.finditer(transcript or ""):
        name = _clean_name(m.group(1) or m.group(2) or "")
        if not name:
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        # отсекаем слишком длинные/короткие куски
        if 2 <= len(name.split()) <= 3 and len(name) <= 40:
            found.append(name)
    return found


def _enrich_owners(report, transcript):
    owners = _extract_owners_from_text(transcript)
    if not owners:
        return report
    assigned = set()
    for t in report.get("tasks") or []:
        o = (t.get("owner") or "").strip()
        if o:
            assigned.add(o.lower())
    # Чистим уже извлечённые LLM имена (убираем ведущие неименные слова).
    for t in report.get("tasks") or []:
        o = (t.get("owner") or "").strip()
        if o:
            t["owner"] = _clean_name(o)
            assigned.add(t["owner"].lower())
    # Назначаем ещё не распределённые имена задачам без ответственного.
    idx = 0
    for t in report.get("tasks") or []:
        if not (t.get("owner") or "").strip() and idx < len(owners):
            # не дублируем уже назначенное имя
            cand = None
            while idx < len(owners):
                if owners[idx].lower() not in assigned:
                    cand = owners[idx]
                    break
                idx += 1
            if cand:
                t["owner"] = cand
                assigned.add(cand.lower())
        idx += 1
    # Если задач нет вообще, но в речи есть ответственный - создадим задачу-заглушку.
    if not report.get("tasks") and owners:
        report["tasks"] = [{"description": "(ответственный упомянут в речи)", "owner": owners[0], "deadline": None}]
    return report


def _enrich_summary(report, transcript):
    """Fallback для 'итога': если LLM не дала summary, пытаемся извлечь
    предложение(я) с маркерами итога/резюме/выводов либо собрать из задач."""
    summary = (report.get("summary") or "").strip()
    if summary and len(summary) >= 20:
        return report
    sents = []
    for m in _SUMMARY_SENT_RE.finditer(transcript or ""):
        s = m.group(1).strip()
        if s:
            sents.append(s)
    if sents:
        report["summary"] = " ".join(sents)
        return report
    tasks = report.get("tasks") or []
    dls = report.get("deadlines") or []
    if tasks or dls:
        parts = []
        if tasks:
            parts.append("Задачи: " + "; ".join((t.get("description") or "") for t in tasks if t.get("description")))
        if dls:
            parts.append("Сроки: " + "; ".join((d.get("when") or d.get("what") or "") for d in dls if (d.get("when") or d.get("what"))))
    if parts:
        report["summary"] = " ".join(parts)
    return report


# --- Синонимы разделов итогового документа ---
# Каждый канонический раздел и его синонимы (для распознавания заголовков в тексте).
SECTION_SYNONYMS = {
    "summary": ["результат", "результаты", "итог", "итоги", "резюме", "вывод", "выводы",
                "суть", "кратко", "основное", "главное"],
    "meeting": ["встреча", "совещание", "созвон", "заседание", "летучка", "собрание",
                "протокол", "встречи"],
    "tasks": ["задачи", "поручения", "to-do", "todo", "что делать", "действия", "планы",
              "план", "порученное"],
    "deadlines": ["сроки", "дедлайны", "дедлайн", "даты", "крайние сроки", "когда",
                  "срок реализации"],
    "decisions": ["решения", "принятые решения", "договорённости", "договоренности",
                  "постановления", "решили", "договорились"],
}
_BULLET_RE = re.compile(r"^\s*(?:[-*•‣▪]|(?:\d+[.)])\s+)")


def _strip_bullet(line):
    return _BULLET_RE.sub("", line).strip(" \t-–—:")


def _section_header_re():
    """Возвращает {key: regex} для детекции заголовков разделов в начале строки.
    Заголовок = синоним + двоеточие (или строка целиком из синонима)."""
    out = {}
    for key, syns in SECTION_SYNONYMS.items():
        # сортируем по убыванию длины, чтобы 'принятые решения' матчилось раньше 'решения'
        alt = "|".join(re.escape(s) for s in sorted(syns, key=len, reverse=True))
        out[key] = re.compile(
            r"^\s*(?:" + alt + r")\b(?:\s*[:\-–—]\s*|\s*$)", re.IGNORECASE
        )
    return out


def parse_structured_sections(text):
    """Если текст оформлен секциями (заголовок + список), разбирает его напрямую
    в структуру отчёта, не полагаясь на LLM. Возвращает dict или None."""
    if not text or len(text.strip()) < 10:
        return None
    headers = _section_header_re()
    lines = text.splitlines()
    # Находим позиции заголовков
    segments = {}
    current = None
    buf = []
    for ln in lines:
        matched_key = None
        matched = None
        for key, rgx in headers.items():
            m = rgx.match(ln)
            if m:
                matched_key = key
                matched = m
                break
        if matched_key:
            if current is not None:
                segments[current] = buf
            current = matched_key
            buf = []
            # остаток строки после заголовка - это первый пункт раздела
            rest = ln[matched.end():].strip()
            if rest:
                buf.append(rest)
        else:
            if current is not None:
                buf.append(ln)
    if current is not None:
        segments[current] = buf

    if not segments:
        return None

    report = {"summary": "", "tasks": [], "deadlines": [], "decisions": []}
    for key, body in segments.items():
        items = [ _strip_bullet(l) for l in body if l.strip() ]
        items = [i for i in items if i]
        if key == "meeting":
            # Контекст встречи кладём в summary, если там пусто.
            meeting_text = " ".join(items)
            if not report["summary"]:
                report["summary"] = meeting_text
        elif key == "summary":
            report["summary"] = " ".join(items)
        elif key == "tasks":
            for it in items:
                owners = _extract_owners_from_text(it)
                dates = _extract_dates_from_text(it)
                dl = dates[0]["when"] if dates else None
                report["tasks"].append({
                    "description": it,
                    "owner": owners[0] if owners else None,
                    "deadline": dl,
                })
        elif key == "deadlines":
            for it in items:
                dates = _extract_dates_from_text(it)
                if dates:
                    when = dates[0]["when"]
                    what = it.replace(when, "").strip(" \t-–—,.")
                    # убираем висячие предлоги/союзы в начале и конце
                    what = re.sub(r"^(?:до|к|по|для|на|с|от|в|чтобы|чтоб)\b\s*", "", what, flags=re.IGNORECASE).strip(" \t-–—,.")
                    what = re.sub(r"\s*(?:до|к|по|для|на|с|от|в)\s*$", "", what, flags=re.IGNORECASE).strip(" \t-–—,.")
                    report["deadlines"].append({"what": what or it, "when": when})
                else:
                    report["deadlines"].append({"what": it, "when": None})
        elif key == "decisions":
            for it in items:
                report["decisions"].append(it)

    # Если ничего содержательного не извлекли - считаем, что структура не подошла.
    if not report["tasks"] and not report["deadlines"] and not report["decisions"] and not report["summary"]:
        return None
    return report


# Русские названия месяцев (полные и сокращённые) и падежные окончания.
MONTHS_RU = {
    "январ": 1, "янв": 1,
    "феврал": 2, "фев": 2,
    "март": 3, "мар": 3,
    "апрел": 4, "апр": 4,
    "ма": 5, "май": 5,
    "июн": 6,
    "июл": 7,
    "август": 8, "авг": 8,
    "сентябр": 9, "сент": 9, "сен": 9,
    "октябр": 10, "окт": 10,
    "ноябр": 11, "нояб": 11, "ноя": 11,
    "декабр": 12, "дек": 12,
}
WEEKDAYS_RU = {
    "понедельник": 0, "пн": 0,
    "вторник": 1, "вт": 1,
    "сред": 2, "ср": 2,
    "четверг": 3, "чт": 3, "четвер": 3,
    "пятниц": 4, "пт": 4, "пятн": 4,
    "суббот": 5, "сб": 5,
    "воскресень": 6, "вс": 6,
}
_ORDINAL_RE = re.compile(r"^\d{1,2}$")
_NUM_WORDS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5,
    "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10,
    "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18,
    "девятнадцат": 19, "двадцат": 20, "тридцат": 30,
}
# Количественные числительные (для 'через две недели', 'через три дня' и т.п.)
_CARDINAL_WORDS = {
    "один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "двадцать": 20, "тридцать": 30,
}


def _to_int_token(token):
    token = token.strip().lower()
    if _ORDINAL_RE.match(token):
        return int(token)
    if token in _NUM_WORDS:
        return _NUM_WORDS[token]
    # "двадцать пятое" / "двадцать пять" - составные
    parts = token.split()
    if len(parts) == 2 and parts[0] in _NUM_WORDS and parts[1] in _NUM_WORDS:
        return _NUM_WORDS[parts[0]] + _NUM_WORDS[parts[1]]
    return None


def parse_russian_date(text, base_date=None):
    """Преобразует русскоязычное указание даты в ISO-строку 'YYYY-MM-DD' или None.

    Поддерживает:
      - '15 октября 2026 года' / '15 октября 2026 г.' / '15 октября 2026'
      - '15 октября' (без года -> текущий/следующий год)
      - '15.10.2026', '15/10/2026', '15-10-2026'
      - '15.10', '15/10'
      - 'в пятницу', 'в следующий понедельник', 'во вторник'
      - 'через неделю', 'через две недели', 'через 3 дня'
      - 'завтра', 'послезавтра', 'сегодня'
    """
    if not text:
        return None
    if base_date is None:
        base_date = date.today()
    s = text.lower()

    # 1) явные цифровые форматы ДД.ММ.ГГГГ / ДД.ММ
    m = re.search(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b", s)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = m.group(3)
        if y:
            y = int(y)
            if y < 100:
                y += 2000
        else:
            y = base_date.year
            # если дата уже прошла в этом году - берём следующий год
            if (mo < base_date.month) or (mo == base_date.month and d < base_date.day):
                y += 1
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass

    # 2) 'число месяц(а) год'
    day = None
    month = None
    year = None
    # число
    dm = re.search(r"\b(\d{1,2}|" + "|".join(_NUM_WORDS.keys()) + r")\b", s)
    if dm:
        day = _to_int_token(dm.group(1))
    # месяц
    for key, num in MONTHS_RU.items():
        if re.search(r"\b" + re.escape(key) + r"[а-яё]*\b", s):
            month = num
            break
    # год
    ym = re.search(r"\b(20\d{2})\b|\b(\d{2})\s*год", s)
    if ym:
        year = int(ym.group(1) or ym.group(2))
        if year < 100:
            year += 2000
    if day and month:
        if not year:
            year = base_date.year
            if (month < base_date.month) or (month == base_date.month and day < base_date.day):
                year += 1
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    # 3) относительные дни
    if re.search(r"\bзавтра\b", s):
        return (base_date + __import__("datetime").timedelta(days=1)).isoformat()
    if re.search(r"\bпослезавтра\b", s):
        return (base_date + __import__("datetime").timedelta(days=2)).isoformat()
    if re.search(r"\bсегодня\b", s):
        return base_date.isoformat()

    # 4) 'через N недель/дней'
    _card_keys = "|".join(list(_CARDINAL_WORDS.keys()) + list(_NUM_WORDS.keys()))
    rel = re.search(
        r"через\s+(?:(\d{1,2}|" + _card_keys + r")\s+)?(недел|дн|недели|неделю)",
        s,
    )
    if rel:
        raw = rel.group(1)
        if raw and raw.isdigit():
            n = int(raw)
        elif raw:
            n = _CARDINAL_WORDS.get(raw) or _to_int_token(raw) or 1
        else:
            n = 1
        unit = rel.group(2)
        days = n * 7 if unit.startswith("недел") else n
        return (base_date + __import__("datetime").timedelta(days=days)).isoformat()

    # 5) дни недели (по основам, чтобы ловить все падежи: пятницу/пятницы/...)
    for wd_name, wd in WEEKDAYS_RU.items():
        if re.search(re.escape(wd_name), s):
            days_ahead = (wd - base_date.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            if re.search(r"прошл", s):
                days_ahead = -((base_date.weekday() - wd) % 7 or 7)
            elif re.search(r"(след|будущ|на будущ)", s):
                if days_ahead == 0:
                    days_ahead = 7
            return (base_date + __import__("datetime").timedelta(days=days_ahead)).isoformat()

    return None


# Регулярка для поиска фраз-дат в тексте (для fallback-извлечения).
_DATE_PHRASE_RE = re.compile(
    r"("
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"                      # 15.10.2026
    r"|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"(?:\s+20\d{2}\s*г(?:ода|г)?\.?|\s+20\d{2}|\s*г(?:ода|г)?\.?)?"  # 15 октября 2026 года
    r"|\b(?:завтра|послезавтра|сегодня)\b"
    r"|через\s+(?:\d{1,2}\s+|" + "|".join(_CARDINAL_WORDS.keys()) + r"\s+)?(?:неделю|недели|недель|дней|дня)"
    r"|(?:в\s+|до\s+|на\s+)?(?:следующ|будущ|прошл)[а-яё]*\s+(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)[а-яё]*"
    r"|(?:в\s+|до\s+|на\s+)?(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)[а-яё]*"
    r")",
    re.IGNORECASE,
)


def _sentences(text):
    return [s.strip() for s in re.split(r"[.!?…]+", text) if s.strip()]


def _extract_dates_from_text(transcript, base_date=None):
    """Fallback: находит в расшифровке все фразы-даты и возвращает
    список {'what','when','when_iso'} по предложению, где встречена дата."""
    found = []
    seen = set()
    for sent in _sentences(transcript):
        for m in _DATE_PHRASE_RE.finditer(sent):
            phrase = m.group(0).strip()
            iso = parse_russian_date(phrase, base_date)
            key = (phrase.lower(), iso)
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "what": sent[:120],
                "when": phrase,
                "when_iso": iso,
            })
    return found


def _norm_field(value, base_date):
    if not value:
        return value, None
    iso = parse_russian_date(value, base_date)
    return value, iso


def normalize_dates_in_report(report, transcript=None, base_date=None):
    """Нормализует все упоминания дат в отчёте в ISO и, при необходимости,
    добавляет пропущенные сроки из расшифровки (fallback)."""
    if base_date is None:
        base_date = date.today()

    for t in report.get("tasks") or []:
        dl = t.get("deadline")
        if dl:
            _, iso = _norm_field(dl, base_date)
            if iso:
                t["deadline_iso"] = iso

    existing = []
    for d in report.get("deadlines") or []:
        when = d.get("when")
        _, iso = _norm_field(when, base_date)
        if iso:
            d["when_iso"] = iso
        if when:
            existing.append(when.lower())

    # Fallback: если LLM вообще не выдала сроков, но в речи они есть.
    if not report.get("deadlines") and transcript:
        for d in _extract_dates_from_text(transcript, base_date):
            report.setdefault("deadlines", []).append(d)

    # Fallback: ответственные (по синонимам) и итог (по синонимам).
    report = _enrich_owners(report, transcript)
    report = _enrich_summary(report, transcript)
    return report


class AnalyzeError(Exception):
    pass


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        # убираем markdown-обёртку ```json ... ```
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _ensure_llm_model(model_name, model_url, progress_cb=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    if progress_cb:
        progress_cb(0.0, "Загрузка локальной LLM...")
    tmp_path = model_path + ".part"
    with requests.get(model_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(min(0.99, done / total), "Загрузка локальной LLM...")
    os.replace(tmp_path, model_path)
    if progress_cb:
        progress_cb(1.0, "Модель загружена")
    return model_path


def analyze_local(
    transcript,
    model_name=DEFAULT_LLM_MODEL,
    model_url=DEFAULT_LLM_URL,
    progress_cb=None,
    n_threads=None,
):
    """Локальный анализ встречи через llama.cpp (GGUF). Полностью оффлайн."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise AnalyzeError(
            "Не установлен llama-cpp-python. Выполните: uv pip install llama-cpp-python"
        )
    model_path = _ensure_llm_model(model_name, model_url, progress_cb)
    if progress_cb:
        progress_cb(1.0, "Инициализация LLM...")
    llm = Llama(
        model_path=model_path,
        n_ctx=LLM_CTX,
        n_gpu_layers=0,
        n_threads=n_threads or max(1, (os.cpu_count() or 4) // 2),
        verbose=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Расшифровка встречи:\n\n{transcript}"},
    ]
    res = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=3072,
        repeat_penalty=1.15,
    )
    text = res["choices"][0]["message"]["content"]
    return _parse_json(text)


def analyze_openai(transcript, api_key, model="gpt-4o-mini"):
    if not api_key:
        raise AnalyzeError("Не указан OpenAI api_key.")
    body = {
        "model": model,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Расшифровка встречи:\n\n{transcript}",
            },
        ],
    }
    resp = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=180,
    )
    if resp.status_code != 200:
        raise AnalyzeError(f"OpenAI вернул {resp.status_code}: {resp.text}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return _parse_json(text)


def analyze(transcript, provider, **kwargs):
    # Если текст оформлен секциями (Задачи/Сроки/Результат/... в любых синонимах),
    # разбираем его напрямую - это точнее и не зависит от размера LLM.
    structured = parse_structured_sections(transcript)
    if structured:
        structured.setdefault("summary", "")
        structured.setdefault("tasks", [])
        structured.setdefault("deadlines", [])
        structured.setdefault("decisions", [])
        return normalize_dates_in_report(structured, transcript)

    if provider == "openai":
        report = analyze_openai(transcript, kwargs.get("openai_key"), kwargs.get("openai_model", "gpt-4o-mini"))
    else:
        report = analyze_local(
            transcript,
            kwargs.get("llm_model", DEFAULT_LLM_MODEL),
            kwargs.get("llm_model_url", DEFAULT_LLM_URL),
            progress_cb=kwargs.get("progress_cb"),
        )
    report.setdefault("summary", "")
    report.setdefault("tasks", [])
    report.setdefault("deadlines", [])
    report.setdefault("decisions", [])
    return normalize_dates_in_report(report, transcript)
