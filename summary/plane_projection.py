"""Plane projections from the accepted effective document; no inference or ASR.

Only the application renderer supplies HTML. Model/source strings remain escaped.
Plane pages receive the agreed eight sections, never private runtime artifacts.
"""
from __future__ import annotations

import html
from html.parser import HTMLParser
import re
from urllib.parse import urlsplit


class WikiHTML(HTMLParser):
    """Small portable HTML subset; retain disclosure contents as visible text.

    Plane editor versions vary in disclosure support. This projection expands
    details rather than silently losing them. App UI/download keeps disclosures.
    """
    allowed = {'h1','h2','h3','h4','p','ul','ol','li','strong','em','br','a','blockquote'}

    def __init__(self, transcript_url):
        super().__init__(convert_charrefs=True)
        self.url = transcript_url
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'summary':
            self.parts.append('<p><strong>')
        elif tag in self.allowed:
            suffix = ''
            if tag == 'a':
                href = dict(attrs).get('href', '')
                if href.startswith('transcript.html#u-'):
                    href = self.url + '#' + href.split('#', 1)[1]
                u = urlsplit(href)
                if u.scheme in ('http', 'https') and u.netloc and not u.username and not u.password:
                    suffix = ' href="' + html.escape(href, quote=True) + '"'
            self.parts.append('<' + tag + suffix + '>')

    def handle_endtag(self, tag):
        if tag == 'summary':
            self.parts.append('</strong></p>')
        elif tag in self.allowed and tag != 'br':
            self.parts.append('</' + tag + '>')

    def handle_data(self, data):
        self.parts.append(html.escape(data))


def app_origin(value):
    p = urlsplit(value)
    if p.scheme not in ('https','http') or not p.netloc or p.username or p.password or p.path or p.query or p.fragment:
        raise ValueError('Задайте адрес приложения для ссылок Plane')
    return value


def project_view(view, job_id, origin, hypothesis_ids, source_index=None):
    """Same tasks, manual overrides and review notices as the summary UI."""
    origin = app_origin(origin)
    doc = view.rendered['summary.json']
    transcript_url = f'{origin}/result?id={int(job_id)}'
    summary_url = f'{origin}/summary?id={int(job_id)}'
    esc = lambda text: html.escape(str(text), quote=True)
    def sources(ids):
        return ', '.join(f'<a href="{esc(transcript_url)}#u-{esc(uid)}">{esc(uid)}</a>' for uid in dict.fromkeys(ids))
    items = []
    status_labels = {'proposed':'Предложено / нужно распределить','committed':'Принято в обсуждении',
                     'in_progress':'В работе по словам участника','unknown':'Не установлен'}
    for task in view.tasks:
        description = f'<p>{esc(task["description"])}</p><p><strong>Исполнитель по конспекту:</strong> {esc(task.get("assignee") or "Не назначен")}</p>'
        if len(task['title']) > 255:
            description = '<p><strong>Полное название:</strong> ' + esc(task['title']) + '</p>' + description
        description += '<p><strong>Статус по обсуждению:</strong> ' + esc(status_labels.get(task.get('discussion_status'), 'Не установлен')) + '</p>'
        for field, label in [('due','Срок по конспекту'),('priority','Приоритет по конспекту'),('recipient','Получатель')]:
            if task.get(field):
                description += f'<p><strong>{label}:</strong> {esc(task[field])}</p>'
        description += f'<p><strong>Источник:</strong> {sources(task["source_ids"])}</p>'
        description += f'<p><a href="{esc(summary_url)}">Конспект встречи</a></p>'
        items.append({'kind':'task','item_id':task['action_id'],'title':task['title'][:255],
                      'description':task['description'],'description_html':description})
    ideas = doc['ideas']
    if len(ideas) != len(hypothesis_ids):
        raise ValueError('Не совпали identity гипотез')
    for idea, identifier in zip(ideas, hypothesis_ids):
        # A short caption is a display label; the full original idea is retained.
        title = 'Гипотеза: ' + idea['text'][:230]
        description = '<p><strong>Предложение для проверки. Не принятое обязательство.</strong></p>'
        description += f'<p>{esc(idea["text"])}</p><p><strong>Источник:</strong> {sources(idea["source_ids"])}</p>'
        description += f'<p><a href="{esc(summary_url)}">Конспект встречи</a></p>'
        items.append({'kind':'hypothesis','item_id':identifier,'title':title,
                      'description':idea['text'],'description_html':description})
    from summary.plane_wiki import clean_summary, participant_header, transcript_block
    parser = WikiHTML(transcript_url)
    fragment = view.rendered['summary.fragment.html']
    if source_index is not None:
        if source_index['source_sha256'] != view.source_sha256:
            raise ValueError('Транскрипция не соответствует конспекту')
        generated_title = re.search(r'<h1>(.*?)</h1>', fragment, re.S)
        fragment = clean_summary(fragment, meeting_title=html.unescape(generated_title.group(1)) if generated_title else None, keep_timecodes=True)
    parser.feed(fragment)
    title_match = re.search(r'<h1>(.*?)</h1>', view.rendered['summary.fragment.html'], re.S)
    title = html.unescape(title_match.group(1)) if title_match else 'Встреча'
    page = {'title':title[:255], 'description_html':
            (participant_header(source_index) + transcript_block(source_index, transcript_url) if source_index is not None else '') +
            f'<p><a href="{esc(summary_url)}">Открыть конспект и карточки в TranscriSummaryzator</a></p>' + ''.join(parser.parts)}
    return items, page


def add_action_controls(fragment, tasks, hypothesis_ids):
    """Add controls only to live UI, without changing sealed exports."""
    for task in tasks:
        opening = '<section class="summary-task" data-action-id="' + html.escape(task['action_id'], quote=True) + '">'
        start = fragment.find(opening)
        end = fragment.find('</section>', start) if start >= 0 else -1
        if end >= 0:
            control = '<div class="plane-action" data-plane-kind="task" data-plane-item-id="' + html.escape(task['action_id'], quote=True) + '"></div>'
            fragment = fragment[:end] + control + fragment[end:]
    for number, identifier in enumerate(hypothesis_ids, 1):
        marker = f'<li><strong>H-{number:02d}.</strong>'
        start = fragment.find(marker)
        end = fragment.find('</li>', start) if start >= 0 else -1
        if end >= 0:
            control = '<div class="plane-action" data-plane-kind="hypothesis" data-plane-item-id="' + html.escape(identifier, quote=True) + '"></div>'
            fragment = fragment[:end] + control + fragment[end:]
    return fragment
