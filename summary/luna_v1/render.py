"""Pure exports of one effective summary document and one source index.

No Markdown or HTML supplied by the model is interpreted as markup. Source
links are constructed exclusively from validated local transcript times.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import html
import json

from .contract import validate_document


HEADINGS = (
    "Главное",
    "Таймкоды",
    "Задачи и следующие шаги",
    "Что осталось уточнить",
    "Технические выводы и ограничения",
    "Идеи и эксперименты, ещё не проверенные",
    "Требует проверки источника",
    "Подробная хронология встречи",
)
_EMPTY = "Сведения для раздела не извлечены или не подтверждены по источнику."
_TASK_FIELDS = (
    "action_id", "title", "description", "discussion_status", "assignee",
    "due", "priority", "recipient", "source_ids", "field_sources", "revision",
)
_STATUS = {
    "proposed": "Предложено / нужно распределить",
    "committed": "Принято в обсуждении",
    "in_progress": "В работе по словам участника",
    "unknown": "Не установлен",
}


def _plain(value: object) -> str:
    return " ".join(str(value).split())


def _md(value: object) -> str:
    text = html.escape(_plain(value), quote=False)
    text = text.replace("\\", "\\\\")
    for char in "`*_[]()#|!":
        text = text.replace(char, "\\" + char)
    return text


def _html(value: object) -> str:
    return html.escape(_plain(value), quote=True)


def _stamp(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _anchor(source_id: str, index: dict) -> tuple[str, str]:
    record = index["by_id"][source_id]
    return f"transcript.html#t-{record['start_ms']}", f"{source_id} {_stamp(record['start_ms'])}"


def _links(ids: list[str], index: dict) -> tuple[str, str]:
    unique = list(dict.fromkeys(ids))
    md_links, html_links = [], []
    for source_id in unique:
        href, label = _anchor(source_id, index)
        md_links.append(f"[{label}]({href})")
        html_links.append(f'<a href="{href}">{_html(label)}</a>')
    return ", ".join(md_links), ", ".join(html_links)


def _navigation_label(item: dict, index: dict) -> tuple[str, str]:
    start = _stamp(index["by_id"][item["start_id"]]["start_ms"])
    end_id = item["end_id"]
    if end_id is not None:
        end = _stamp(index["by_id"][end_id]["end_ms"])
        if end != start:
            return f"{start}–{end}", _anchor(item["start_id"], index)[0]
    return start, _anchor(item["start_id"], index)[0]


def _title(document: dict, index: dict) -> str:
    meeting_date = index.get("meeting_date")
    if meeting_date:
        try:
            label = date.fromisoformat(meeting_date).strftime("%d.%m.%Y")
        except ValueError:
            raise ValueError("source index has invalid meeting date") from None
    else:
        label = "Встреча"
    project = document["meeting"]["project"]
    topic = document["meeting"]["topic"]
    return f"{label} | {project} — {topic}" if project else f"{label} — {topic}"


def _effective_tasks(document: dict, index: dict, overrides: list[dict] | None) -> list[dict]:
    supplied = document["tasks"] if overrides is None else overrides
    if not isinstance(supplied, list) or len(supplied) != len(document["tasks"]):
        raise ValueError("effective task count differs from generated document")
    result = []
    first_source_ordinals: dict[str, int] = {}
    for number, (generated, item) in enumerate(zip(document["tasks"], supplied), 1):
        if not isinstance(item, dict):
            raise ValueError(f"effective task {number} is not an object")
        task = deepcopy(generated)
        for field in _TASK_FIELDS:
            if field in item:
                task[field] = deepcopy(item[field])
        if task["source_ids"] != generated["source_ids"]:
            raise ValueError("task override cannot change sealed source coordinates")
        if task["field_sources"] != generated["field_sources"]:
            raise ValueError("task override cannot change sealed field evidence")
        for field in ("title", "description"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise ValueError(f"effective task {number} has empty {field}")
        for field in ("assignee", "due", "priority", "recipient"):
            if task[field] is not None and not isinstance(task[field], str):
                raise ValueError(f"effective task {number} has invalid {field}")
        if task["discussion_status"] not in _STATUS:
            raise ValueError(f"effective task {number} has invalid status")
        if not task.get("action_id"):
            # A deterministic initial key; the backend must persist it and
            # pass it back as an override across edits and regenerations.
            first = task["source_ids"][0]
            first_source_ordinals[first] = first_source_ordinals.get(first, 0) + 1
            material = f"{index['source_sha256']}:{first}:{first_source_ordinals[first]}"
            task["action_id"] = "A-" + hashlib.sha256(material.encode()).hexdigest()[:20]
        if not isinstance(task["action_id"], str) or not task["action_id"].strip():
            raise ValueError(f"effective task {number} has invalid action ID")
        if "revision" not in task:
            task["revision"] = 0
        result.append({field: task[field] for field in _TASK_FIELDS if field in task})
    if len({item["action_id"] for item in result}) != len(result):
        raise ValueError("effective tasks have duplicate action IDs")
    return result


def _transcript_turns(index: dict, include_ids: set[str] | None = None) -> str:
    turns = []
    anchored: set[int] = set()
    for source_id, row in index["by_id"].items():
        if include_ids is not None and source_id not in include_ids:
            continue
        milliseconds = row["start_ms"]
        anchor = f' id="t-{milliseconds}"' if milliseconds not in anchored else ""
        anchored.add(milliseconds)
        caution = " <small>Неясная реплика в готовой транскрипции</small>" if row.get("needs_review") else ""
        turns.append(
            f'<section class="turn"{anchor}><time>{_stamp(milliseconds)}</time> '
            f'<strong>{_html(row["speaker"])}</strong> '
            f'<span>{_html(row["text"])}</span>{caution}</section>'
        )
    return "\n".join(turns)


def _cited_ids(document: dict) -> set[str]:
    ids: set[str] = set()
    for field in ("main", "tasks", "questions", "technical", "ideas", "verification", "chapters"):
        for item in document[field]:
            ids.update(item["source_ids"])
            if field == "chapters":
                for detail in item["details"]:
                    ids.update(detail["source_ids"])
    for field in ("timecodes", "chapters"):
        for item in document[field]:
            ids.add(item["start_id"])
            if item["end_id"] is not None:
                ids.add(item["end_id"])
    return ids


def _transcript_html(index: dict) -> str:
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Транскрипция</title>'
        '<style>body{font:17px/1.55 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}'
        '.turn{padding:.8rem 0;border-bottom:1px solid #ddd;scroll-margin-top:1rem}'
        'time{font:14px monospace;color:#555;margin-right:.5rem}small{color:#955;margin-left:.4rem}'
        '</style></head><body><h1>Транскрипция</h1>'
        + _transcript_turns(index) + '</body></html>'
    )


def render_document(document: dict, source_index: dict, effective_tasks: list[dict] | None = None,
                    quality_review: dict | None = None) -> dict:
    """Return md/html/json/tasks built from a single effective state.

    ``effective_tasks`` is the backend's versioned projection of user edits.
    It must carry the persistent action IDs when present; this renderer never
    writes overrides, changes the sealed model document, or calls a model.
    """
    validate_document(document, source_index)
    tasks = _effective_tasks(document, source_index, effective_tasks)
    effective = deepcopy(document)
    effective["tasks"] = deepcopy(tasks)
    effective["meeting"]["date"] = source_index.get("meeting_date")
    effective["meeting"]["participants_by_transcript"] = list(source_index.get("participants", []))
    effective["meeting"]["unattributed_speech"] = bool(source_index.get("unattributed_speech"))
    effective["source_sha256"] = source_index["source_sha256"]
    if quality_review is not None:
        effective["quality_review"] = deepcopy(quality_review)

    md: list[str] = []
    fragment: list[str] = []
    title = _title(document, source_index)
    md.extend([f"# {_md(title)}", ""])
    fragment.append(f"<h1>{_html(title)}</h1>")
    people = list(source_index.get("participants", []))
    if source_index.get("unattributed_speech"):
        people.append("Участник не определён")
    participants = ", ".join(people) if people else "Не определены"
    md.extend([f"**Участники по транскрипции:** {_md(participants)}", ""])
    fragment.append(f"<p><strong>Участники по транскрипции:</strong> {_html(participants)}</p>")
    if quality_review is not None and quality_review.get("status") != "checked":
        status = quality_review.get("status")
        count = quality_review.get("unresolved_count", 0)
        if status == "unresolved":
            notice = f"Автоматическая проверка оставила {count} вопрос(ов); они отмечены в разделе «Требует проверки источника»."
        elif status == "postverify_corrected_unchecked":
            notice = "После итоговой проверки внесено ещё одно адресное исправление; локально проверены его формат и ссылки, но смысл повторно не проверялся Luna."
            if count:
                notice += f" Осталось {count} вопрос(ов) в разделе «Требует проверки источника»."
        else:
            notice = "Автоматическая смысловая проверка завершилась не полностью; конспект опубликован с этой пометкой."
        md.extend([f"**Качество конспекта:** {_md(notice)}", ""])
        fragment.append(f"<p class=\"summary-quality-notice\"><strong>Качество конспекта:</strong> {_html(notice)}</p>")

    def heading(text: str) -> None:
        md.extend([f"## {text}", ""])
        fragment.append(f"<h2>{_html(text)}</h2>")

    def empty() -> None:
        md.extend([_EMPTY, ""])
        fragment.append(f"<p>{_html(_EMPTY)}</p>")

    heading(HEADINGS[0])
    if document["main"]:
        for item in document["main"]:
            source_md, source_html = _links(item["source_ids"], source_index)
            md.extend([f"{_md(item['text'])} ({source_md})", ""])
            fragment.append(f"<p>{_html(item['text'])} <small>({source_html})</small></p>")
    else:
        empty()

    heading(HEADINGS[1])
    if document["timecodes"]:
        fragment.append("<ul>")
        for item in document["timecodes"]:
            label, href = _navigation_label(item, source_index)
            md.append(f"- [{label}]({href}) — {_md(item['topic'])}")
            fragment.append(f'<li><a href="{href}">{_html(label)}</a> — {_html(item["topic"])}</li>')
        md.append("")
        fragment.append("</ul>")
    else:
        empty()

    heading(HEADINGS[2])
    if tasks:
        for number, task in enumerate(tasks, 1):
            md.extend([f"### T-{number:02d}. {_md(task['title'])}", "", f"**Исполнитель:** {_md(task['assignee'] or 'Не назначен')}", "", f"**Описание:** {_md(task['description'])}", ""])
            fragment.extend([f"<section class=\"summary-task\" data-action-id=\"{_html(task['action_id'])}\"><h3>T-{number:02d}. {_html(task['title'])}</h3>",
                             f"<p><strong>Исполнитель:</strong> {_html(task['assignee'] or 'Не назначен')}</p>",
                             f"<p><strong>Описание:</strong> {_html(task['description'])}</p>"])
            if task["discussion_status"] != "unknown":
                md.extend([f"**Статус по обсуждению:** {_md(_STATUS[task['discussion_status']])}", ""])
                fragment.append(f"<p><strong>Статус по обсуждению:</strong> {_html(_STATUS[task['discussion_status']])}</p>")
            for field, caption in (("due", "Срок"), ("priority", "Приоритет"), ("recipient", "Получатель")):
                if task[field] is not None:
                    md.extend([f"**{caption}:** {_md(task[field])}", ""])
                    fragment.append(f"<p><strong>{caption}:</strong> {_html(task[field])}</p>")
            source_md, source_html = _links(task["source_ids"], source_index)
            md.extend([f"**Источник:** {source_md}", ""])
            fragment.extend([f"<p><strong>Источник:</strong> {source_html}</p>", "</section>"])
    else:
        empty()

    for heading_text, field, prefix in (
        (HEADINGS[3], "questions", "Q"),
        (HEADINGS[4], "technical", "X"),
        (HEADINGS[5], "ideas", "H"),
    ):
        heading(heading_text)
        rows = document[field]
        if not rows:
            empty()
            continue
        fragment.append("<ul>")
        for number, item in enumerate(rows, 1):
            source_md, source_html = _links(item["source_ids"], source_index)
            md.append(f"- **{prefix}-{number:02d}.** {_md(item['text'])} ({source_md})")
            fragment.append(f"<li><strong>{prefix}-{number:02d}.</strong> {_html(item['text'])} <small>({source_html})</small></li>")
        md.append("")
        fragment.append("</ul>")

    heading(HEADINGS[6])
    if document["verification"]:
        fragment.append("<ul>")
        for number, item in enumerate(document["verification"], 1):
            source_md, source_html = _links(item["source_ids"], source_index)
            md.append(f"- **V-{number:02d}.** {_md(item['text'])} — {_md(item['why_unresolved'])} ({source_md})")
            fragment.append(f"<li><strong>V-{number:02d}.</strong> {_html(item['text'])} — {_html(item['why_unresolved'])} <small>({source_html})</small></li>")
        md.append("")
        fragment.append("</ul>")
    else:
        empty()

    heading(HEADINGS[7])
    if document["chapters"]:
        for item in document["chapters"]:
            label, href = _navigation_label(item, source_index)
            source_md, source_html = _links(item["source_ids"], source_index)
            md.extend([f"### [{label}]({href}) — {_md(item['topic'])}", "", f"{_md(item['summary'])} ({source_md})", "", "<details>", "<summary>Детали по источнику</summary>", ""])
            fragment.extend([f'<section class="summary-chapter"><h3><a href="{href}">{_html(label)}</a> — {_html(item["topic"])}</h3>',
                             f"<p>{_html(item['summary'])} <small>({source_html})</small></p>",
                             "<details><summary>Детали по источнику</summary>"])
            if item["details"]:
                fragment.append("<ul>")
                for detail in item["details"]:
                    detail_md, detail_html = _links(detail["source_ids"], source_index)
                    md.append(f"- {_md(detail['text'])} ({detail_md})")
                    fragment.append(f"<li>{_html(detail['text'])} <small>({detail_html})</small></li>")
                fragment.append("</ul>")
            else:
                md.append(_EMPTY)
                fragment.append(f"<p>{_html(_EMPTY)}</p>")
            md.extend(["", "</details>", ""])
            fragment.extend(["</details>", "</section>"])
    else:
        empty()

    body = "\n".join(fragment)
    # A standalone download and the dashboard fragment share exactly this body.
    standalone_body = body.replace('href="transcript.html#', 'href="#')
    standalone_body += (
        '<details><summary>Исходные реплики для ссылок</summary>'
        + _transcript_turns(source_index, _cited_ids(document)) + '</details>'
    )
    standalone = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{_html(title)}</title>'
        '<style>body{font:17px/1.55 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}'
        'h2{margin-top:2rem}h3{margin-top:1.5rem}li{margin:.5rem 0}details{margin:1rem 0;padding:.5rem;border:1px solid #bbb}'
        'a{color:#0758a8}small{color:#555}.summary-task{padding:.5rem 0;border-bottom:1px solid #ddd}</style>'
        '</head><body>' + standalone_body + '</body></html>'
    )
    return {
        "summary.md": "\n".join(md).rstrip() + "\n",
        "summary.html": standalone,
        "summary.fragment.html": body,
        "summary.json": effective,
        "tasks.json": deepcopy(tasks),
        "transcript.html": _transcript_html(source_index),
    }
