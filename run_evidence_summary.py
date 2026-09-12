#!/usr/bin/env python3
"""Evidence-first meeting summary benchmark for local Ollama models."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


LINE_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d(?:\.\d+)?)\]\s+([^:]+):\s*(.*)$")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class Status:
    def __init__(self, path: Path, run: str, model: str, output_dir: Path):
        self.path = path
        self.started = time.time()
        self.data = {
            "run": run,
            "model": model,
            "status": "running",
            "phase": "Подготовка",
            "progress": 0.0,
            "detail": "Запускаю тест",
            "started_at": iso_now(),
            "updated_at": iso_now(),
            "elapsed_seconds": 0,
            "generated_tokens": 0,
            "output_dir": str(output_dir),
        }
        self.write()

    def update(self, progress, phase, detail, **extra):
        self.data.update(extra)
        self.data.update({
            "progress": round(max(0, min(100, progress)), 1),
            "phase": phase,
            "detail": detail,
            "updated_at": iso_now(),
            "elapsed_seconds": int(time.time() - self.started),
        })
        self.write()

    def write(self):
        atomic_json(self.path, self.data)

    def fail(self, exc):
        self.data.update({"status": "failed", "phase": "Ошибка", "detail": str(exc), "updated_at": iso_now()})
        self.write()

    def done(self, summary: Path):
        self.update(100, "Готово", "Саммари проверено и сохранено", status="done", finished_at=iso_now(), summary_path=str(summary))


def ollama_chat(model, system, prompt, status=None, progress_range=(0, 100), expected_tokens=2500, temperature=0.1, json_format=False):
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "top_p": 0.85, "repeat_penalty": 1.08, "num_ctx": 32768, "num_predict": max(1200, expected_tokens)},
    }
    if json_format:
        payload["format"] = "json"
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    parts, tokens, final = [], 0, {}
    base_tokens = int(status.data.get("generated_tokens", 0)) if status else 0
    start, end = progress_range
    last_write = 0.0
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            event = json.loads(raw)
            content = event.get("message", {}).get("content", "")
            if content:
                parts.append(content)
                tokens += 1
            if event.get("done"):
                final = event
            now = time.time()
            if status and now - last_write >= 1:
                fraction = min(0.96, tokens / max(1, expected_tokens))
                status.update(start + (end - start) * fraction, status.data["phase"], f"Сгенерировано токенов: {tokens}", generated_tokens=base_tokens + tokens)
                last_write = now
    return "".join(parts), final


def parse_transcript(path: Path):
    utterances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        hours, minutes, seconds, speaker, text = match.groups()
        start = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        utterances.append({"id": f"U{len(utterances)+1:05d}", "time": f"{hours}:{minutes}:{float(seconds):06.3f}", "start": start, "speaker": speaker.strip(), "text": text.strip()})
    if not utterances:
        raise ValueError("В расшифровке не найдены реплики с таймкодами")
    return utterances


def chunks(utterances, max_chars=18000):
    result, current, size = [], [], 0
    for item in utterances:
        line = f'{item["id"]} [{item["time"]}] {item["speaker"]}: {item["text"]}'
        if current and size + len(line) > max_chars:
            result.append(current)
            current, size = [], 0
        current.append((item, line))
        size += len(line) + 1
    if current:
        result.append(current)
    return result


def extract_json(text):
    candidate = text.strip()
    match = JSON_FENCE_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    start = min([position for position in (candidate.find("["), candidate.find("{")) if position >= 0], default=-1)
    if start > 0:
        candidate = candidate[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        end = max(candidate.rfind("]"), candidate.rfind("}"))
        if end >= 0:
            return json.loads(candidate[: end + 1])
        raise


EXTRACT_SYSTEM = """Ты — строгий аналитик стенограмм. Извлекай только явно произнесённые факты. Не додумывай решения, даты, ответственных, расшифровки аббревиатур и причинно-следственные связи. Реплики содержат стабильные ID; каждое утверждение обязано ссылаться на них. Верни только JSON без Markdown."""


def extract_facts(model, utterances, status, output_dir):
    all_facts = []
    groups = chunks(utterances)
    speakers = sorted({item["speaker"] for item in utterances})
    for index, group in enumerate(groups, 1):
        status.update(5 + 43 * (index - 1) / len(groups), "Извлечение фактов", f"Часть {index} из {len(groups)}")
        part_path = output_dir / f"facts-part-{index:02d}.json"
        if part_path.is_file():
            cached = json.loads(part_path.read_text(encoding="utf-8"))
            cached_facts = cached.get("facts", [])
            if isinstance(cached_facts, list) and cached_facts:
                all_facts.extend(cached_facts)
                continue
        body = "\n".join(line for _, line in group)
        prompt = f"""Язык встречи: русский. Известные метки участников: {', '.join(speakers)}.
Термины домена сохраняй буквально: SMC, BOS, CHoCH, FVG, TPO, GitLab, Binance, Dow Jones, S&P 500, Nasdaq, имбаланс, swing high, swing low, Stop Loss, Take Profit, Break Even.

Извлеки существенные элементы встречи в JSON-массив. Формат каждого объекта:
{{"kind":"observation|problem|proposal|decision|action|question|metric","statement":"кратко и без домыслов","evidence_ids":["U00001"],"speaker_refs":["@name"],"certainty":"explicit|tentative"}}

Правила:
- decision только если согласие или решение сформулировано явно;
- action только если явно сказано, что сделать; не назначай ответственного без прямых слов;
- числа, даты и параметры копируй точно;
- предложения и гипотезы не превращай в принятые решения;
- 1–4 соседних evidence_ids на факт; не ссылайся на отсутствующие ID;
- пропусти приветствия, повторы и разговорный шум.

СТЕНОГРАММА ЧАСТИ:
{body}"""
        text, metrics = ollama_chat(model, EXTRACT_SYSTEM, prompt, status, (5 + 43*(index-1)/len(groups), 5 + 43*index/len(groups)), 1400, 0.0, json_format=True)
        parsed = extract_json(text)
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else parsed
        if not isinstance(facts, list):
            raise ValueError(f"Модель вернула неверный формат фактов в части {index}")
        for fact in facts:
            if isinstance(fact, dict):
                fact["source_chunk"] = index
                all_facts.append(fact)
        atomic_json(part_path, {"facts": facts, "metrics": metrics})
    return all_facts


def validate_facts(facts, utterances):
    by_id = {item["id"]: item for item in utterances}
    speakers = {item["speaker"] for item in utterances}
    accepted, rejected = [], []
    seen = set()
    for fact in facts:
        ids = [item for item in fact.get("evidence_ids", []) if item in by_id]
        refs = [item for item in fact.get("speaker_refs", []) if item in speakers]
        statement = str(fact.get("statement", "")).strip()
        reason = None
        if not statement or not ids:
            reason = "нет утверждения или действительной ссылки"
        elif len(statement) > 700:
            reason = "слишком длинное утверждение"
        evidence = [by_id[item] for item in ids]
        key = re.sub(r"\W+", " ", statement.casefold()).strip()
        if not reason and key in seen:
            reason = "дубликат"
        if reason:
            rejected.append({"fact": fact, "reason": reason})
            continue
        seen.add(key)
        clean = {
            "fact_id": f"F{len(accepted)+1:04d}",
            "kind": fact.get("kind", "observation"),
            "statement": statement,
            "certainty": fact.get("certainty", "explicit"),
            "speaker_refs": refs,
            "evidence_ids": ids,
            "start": min(item["start"] for item in evidence),
            "end": max(item["start"] for item in evidence),
            "evidence": [f'[{item["time"]}] {item["speaker"]}: {item["text"]}' for item in evidence],
        }
        accepted.append(clean)
    return accepted, rejected


WRITER_SYSTEM = """Ты создаёшь точное русскоязычное саммари встречи только по проверенному реестру фактов. Тип факта является жёстким ограничением: proposal остаётся предложением, observation — наблюдением, problem — проблемой; только decision разрешено назвать принятым решением. Запрещено добавлять сведения из общих знаний, расшифровывать аббревиатуры без доказательства, назначать ответственных, превращать гипотезы в решения и придумывать сроки. При нехватке данных прямо укажи неопределённость. Каждый содержательный абзац или пункт должен иметь таймкод из предоставленного факта."""


def facts_text(facts):
    lines = []
    for fact in facts:
        evidence = " | ".join(fact["evidence"])
        lines.append(f'{fact["fact_id"]} [{fact["kind"]}; {fact["certainty"]}] {fact["statement"]}\nДоказательство: {evidence}')
    return "\n\n".join(lines)


def write_summary(model, facts, status, progress_range):
    prompt = f"""На основе реестра ниже подготовь итоговый документ строго в таком составе:
# Саммари встречи
## 1. Общая информация
- Основная тема
- Цель: если отдельного доказанного факта о цели нет, напиши «Явно не сформулирована»
- Один абзац результата с чётким разделением: принято / предложено / осталось неясным
## 2. Краткое содержание
Хронологический рассказ с таймкодами.
## 3. Основные темы обсуждения
По каждой теме: суть, проблема, предложения, подтверждённый итог, таймкоды.
## 4. Решения
Только explicit decision. Если явных решений мало — так и напиши.
## 5. Задачи и открытые вопросы
Не указывай исполнителя или срок без прямого доказательства.

Формат таймкода: [HH:MM:SS–HH:MM:SS]. Не выводи ID фактов. Не добавляй разделы или факты, отсутствующие в реестре. В блок «Принято» и раздел «Решения» допускаются исключительно записи kind=decision. Записи kind=action описывай как задачи без придуманного исполнителя. Для kind=proposal всегда используй слова «предложено», «обсуждалось» или «гипотеза», даже если формулировка звучит повелительно. Не употребляй «установлено», «согласовано», «решено», «должен» для proposal/observation/problem. Сохраняй термины GitLab, Binance, Dow Jones, S&P 500, Nasdaq, SMC, BOS, CHoCH, FVG, TPO, имбаланс.

ПРОВЕРЕННЫЙ РЕЕСТР ФАКТОВ:
{facts_text(facts)}"""
    return ollama_chat(model, WRITER_SYSTEM, prompt, status, progress_range, 4200, 0.1)


VERIFY_SYSTEM = """Ты — финальный аудитор саммари. Исправляй текст, а не комментируй его. Тип каждого факта является жёстким ограничением: только decision можно назвать принятым, решённым, согласованным или установленным; proposal всегда остаётся предложением или гипотезой; action не получает исполнителя или срок без прямого доказательства. Оставляй только утверждения, поддержанные реестром доказательств. Сохраняй полезную структуру и таймкоды. Удали выдуманные решения, сроки, ответственных, числа, определения и причинность. Не добавляй новых фактов. Верни только исправленный Markdown."""


def verify_summary(verifier_model, draft, facts, status, progress_range):
    prompt = f"""Проверь черновик по реестру. В спорном случае удаляй или маркируй неопределённость. Проверь особенно: в «Принято» и «Решения» могут находиться только kind=decision; kind=proposal нельзя описывать словами «принято», «решено», «согласовано», «установлено» или «должен»; точность чисел; отсутствие назначенных без доказательства ответственных; отсутствие выдуманной расшифровки TPO и других терминов. Если в реестре нет отдельного факта о цели, цель должна быть «Явно не сформулирована».

РЕЕСТР:
{facts_text(facts)}

ЧЕРНОВИК:
{draft}"""
    return ollama_chat(verifier_model, VERIFY_SYSTEM, prompt, status, progress_range, 4300, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--writer-model", required=True)
    parser.add_argument("--extractor-model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--verifier-model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--reuse-facts", type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = Status(args.status, args.run_name, args.writer_model, args.output)
    try:
        utterances = parse_transcript(args.transcript)
        atomic_json(args.output / "input.json", {"transcript": str(args.transcript), "utterances": len(utterances), "speakers": sorted({x['speaker'] for x in utterances})})
        if args.reuse_facts:
            facts = json.loads(args.reuse_facts.read_text(encoding="utf-8"))["facts"]
            status.update(10, "Проверенные факты", f"Использую общий реестр: {len(facts)} фактов")
        else:
            raw = extract_facts(args.extractor_model, utterances, status, args.output)
            status.update(49, "Проверка фактов", "Проверяю ссылки на исходные реплики")
            facts, rejected = validate_facts(raw, utterances)
            atomic_json(args.output / "facts.validated.json", {"facts": facts, "rejected": rejected})
            status.update(55, "Проверка фактов", f"Принято {len(facts)}, отклонено {len(rejected)}")
        start = 12 if args.reuse_facts else 58
        status.update(start, "Создание саммари", f"Пишет {args.writer_model}")
        draft, writer_metrics = write_summary(args.writer_model, facts, status, (start, 82))
        (args.output / "draft.md").write_text(draft.strip() + "\n", encoding="utf-8")
        atomic_json(args.output / "writer.metrics.json", writer_metrics)
        status.update(84, "Финальная проверка", f"Проверяет {args.verifier_model}")
        final, verifier_metrics = verify_summary(args.verifier_model, draft, facts, status, (84, 99))
        summary = args.output / "summary.md"
        summary.write_text(final.strip() + "\n", encoding="utf-8")
        atomic_json(args.output / "verifier.metrics.json", verifier_metrics)
        status.done(summary)
    except Exception as exc:
        status.fail(exc)
        raise


if __name__ == "__main__":
    main()
