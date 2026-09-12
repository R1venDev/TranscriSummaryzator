#!/usr/bin/env python3
import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """Ты — редактор протоколов русскоязычных рабочих встреч. Создай точное, подробное и структурированное саммари только по расшифровке.

Правила:
1. Не выдумывай факты, решения, сроки, ответственных, причины и числовые результаты.
2. Строго различай принятое решение, предложение, гипотезу, наблюдение, открытый вопрос и поручение.
3. Каждый содержательный абзац и крупный пункт сопровождай подтверждающим диапазоном [ЧЧ:ММ:СС–ЧЧ:ММ:СС].
4. Не считай предложение решением без явного согласования.
5. Если часовой пояс, срок, ответственный или значение не названы, укажи это.
6. Исправляй только явные опечатки терминов, не меняя смысл.
7. Пропускай приветствия, шутки и повторы, если они не влияют на решения.
8. Верни только Markdown-документ без вводных замечаний.

Структура:
# Саммари встречи
## 1. Общая информация
- Основная тема
- Цель
Затем 1–2 абзаца главного результата.
## 2. Краткое содержание
Связное хронологическое изложение; каждый абзац заканчивается таймкодом.
## 3. Основные темы обсуждения
Для каждой темы отдельный заголовок третьего уровня. Раскрывай суть, текущее состояние, проблему, предложения, альтернативы, итог и таймкоды, если они есть.
## 4. Принятые решения
Только действительно согласованные решения с таймкодами.
## 5. Задачи и следующие шаги
Действие, ответственный и срок только если следуют из разговора; неизвестные поля — «не указан».
## 6. Открытые вопросы и гипотезы
## 7. Числа, параметры и результаты тестов

Пиши профессионально, подробно и по-русски. Сохраняй общепринятое написание BOS, FBOS, FVG, Order Block, Breaker, TPO, swing, Stop Loss, Take Profit, Break Even, GitLab, Dow Jones, S&P 500, Nasdaq."""

def gpu_monitor(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=5).strip().split(",")
            samples.append({"time": time.time(), "memory_mib": int(out[0]), "utilization_percent": int(out[1])})
        except Exception:
            pass
        stop.wait(0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    transcript = args.transcript.read_text(encoding="utf-8")
    payload = {
        "model": args.model, "stream": False, "think": False, "keep_alive": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "РАСШИФРОВКА ВСТРЕЧИ:\n\n" + transcript}],
        "options": {"num_ctx": 32768, "num_predict": 9000, "temperature": 0.2,
                    "top_p": 0.8, "top_k": 20, "seed": 12072026}}
    (args.output / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stop, samples = threading.Event(), []
    monitor = threading.Thread(target=gpu_monitor, args=(stop, samples), daemon=True)
    monitor.start(); started = time.time()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as response:
            result = json.loads(response.read())
    finally:
        stop.set(); monitor.join(timeout=2)
    wall = time.time() - started
    (args.output / "summary.md").write_text(result.get("message", {}).get("content", "").strip() + "\n", encoding="utf-8")
    (args.output / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ps, es = result.get("prompt_eval_duration", 0)/1e9, result.get("eval_duration", 0)/1e9
    metrics = {
        "model": args.model, "wall_seconds": wall,
        "total_duration_seconds": result.get("total_duration", 0)/1e9,
        "load_duration_seconds": result.get("load_duration", 0)/1e9,
        "prompt_eval_count": result.get("prompt_eval_count"), "prompt_eval_seconds": ps,
        "eval_count": result.get("eval_count"), "eval_seconds": es,
        "prompt_tokens_per_second": result.get("prompt_eval_count", 0)/ps if ps else None,
        "generation_tokens_per_second": result.get("eval_count", 0)/es if es else None,
        "peak_gpu_memory_mib": max((s["memory_mib"] for s in samples), default=None),
        "peak_gpu_utilization_percent": max((s["utilization_percent"] for s in samples), default=None),
        "gpu_samples": samples}
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "gpu_samples"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
