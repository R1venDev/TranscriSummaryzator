"""One native Wiki layout, from immutable source; no semantic/model changes."""
from html import escape
from html.parser import HTMLParser
import re
import json
import uuid

PARTICIPANTS = 'Участники по транскрипции:'


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []
    def text(self):
        return ''.join(c.text() if isinstance(c, Node) else c for c in self.children)
    def html(self):
        inner = ''.join(c.html() if isinstance(c, Node) else escape(c) for c in self.children)
        if not self.tag:
            return inner
        attrs = ''.join(f' {k}="{escape(v or "", quote=True)}"' for k, v in self.attrs.items())
        if self.tag in {'br', 'hr', 'img', 'input'}:
            return f'<{self.tag}{attrs}>'
        return f'<{self.tag}{attrs}>{inner}</{self.tag}>'


class Tree(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(value)
        self.close()
    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'br', 'hr', 'img', 'input'}:
            self.stack.append(node)
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                return
    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean_summary(value, asset_ids=(), media_marker='', transcript_id='', meeting_title=None):
    """Remove generated duplicates, preserving other remote body/edits.

    Only root meeting H1/participant paragraph, exact Timecodes H2 + its list,
    and explicitly identified application assets/blocks are removed.
    """
    tree = Tree(value)
    result, skip_list = [], False
    for node in tree.root.children:
        if not isinstance(node, Node):
            result.append(node)
            continue
        text = node.text().strip()
        if (node.tag == 'h1' and meeting_title is not None and text == meeting_title.strip()) or (node.tag == 'p' and text.startswith(PARTICIPANTS)):
            continue
        if node.tag == 'h2' and text == 'Таймкоды':
            skip_list = True
            continue
        if skip_list:
            if node.tag == 'ul':
                skip_list = False
                continue
            if text:
                skip_list = False
        if node.tag == 'attachment-component' and node.attrs.get('src') in asset_ids:
            continue
        if media_marker and node.attrs.get('data-id') == media_marker:
            continue
        if transcript_id and node.attrs.get('data-id') == transcript_id:
            continue
        result.append(node)
    tree.root.children = result
    return tree.root.html()


def people(index):
    names = list(dict.fromkeys(index.get('participants', [])))
    if index.get('unattributed_speech'):
        names.append('Участник не определён')
    return names or ['Не определены']


def participant_header(index):
    names = people(index)
    metadata = escape(json.dumps(names, ensure_ascii=False), quote=True)
    return '<p data-transcri-participants="' + metadata + '"><strong>' + PARTICIPANTS + '</strong> ' + ', '.join(escape(str(n)) for n in names) + '</p>'


def resolve_mentions(value, members):
    """Unique exact display-name matches only; never fuzzy speaker attribution."""
    def norm(name):
        return str(name).strip().lstrip('@').casefold()
    lookup = {}
    for member in members:
        user = member.get('member', member)
        if not isinstance(user, dict):
            continue
        try:
            identifier = str(uuid.UUID(user['id']))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        name = norm(user.get('display_name', ''))
        if name:
            lookup.setdefault(name, set()).add(identifier)
    tree = Tree(value)
    for node in tree.root.children:
        if not isinstance(node, Node) or node.tag != 'p' or not node.text().strip().startswith(PARTICIPANTS):
            continue
        try:
            names = json.loads(node.attrs['data-transcri-participants'])
        except (KeyError, ValueError, TypeError):
            continue  # legacy display text cannot establish person boundaries
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise ValueError('Неверный список участников')
        header = '<strong>' + PARTICIPANTS + '</strong> '
        rendered, mentioned = [], set()
        for name in names:
            ids = lookup.get(norm(name), set())
            if len(ids) != 1:
                rendered.append(escape(name.strip()))
                continue
            identifier = next(iter(ids))
            if identifier in mentioned:
                continue
            mentioned.add(identifier)
            mention_id = str(uuid.uuid5(uuid.NAMESPACE_URL, 'transcri-mention:' + identifier))
            rendered.append(f'<mention-component id="{mention_id}" entity_identifier="{identifier}" entity_name="user_mention"></mention-component>')
        node.children = Tree(header + ', '.join(rendered)).root.children
        node.attrs.pop('data-transcri-participants', None)
    return tree.root.html()


def transcript_block(index, transcript_url):
    identifier = 'transcri-transcript-' + index['source_sha256'][:32]
    rows = []
    for uid, unit in index['by_id'].items():
        if not re.fullmatch(r'U[0-9]+', uid):
            raise ValueError('Некорректный source ID')
        seconds = int(unit['start_ms']) // 1000
        h, rest = divmod(seconds, 3600)
        m, s = divmod(rest, 60)
        clock = f'{h:02}:{m:02}:{s:02}' if h else f'{m:02}:{s:02}'
        rows.append(f'<p><a href="{escape(transcript_url, quote=True)}#u-{uid}">{clock} · {uid}</a> '
                    f'<strong>{escape(str(unit.get("speaker") or "Не определён"))}:</strong> {escape(unit["text"])}</p>')
    return (f'<details class="editor-details-block" data-id="{identifier}">'
            f'<summary class="editor-details-summary">Исходная транскрипция · {len(rows)} реплик</summary>'
            '<div class="editor-details-content" data-type="detailsContent">' + ''.join(rows) + '</div></details>')


def insert_media(body, blocks):
    """Put media immediately after participant header, before the transcript."""
    tree = Tree(body)
    for i, node in enumerate(tree.root.children):
        if isinstance(node, Node) and node.tag == 'p' and node.text().strip().startswith(PARTICIPANTS):
            tree.root.children[i+1:i+1] = Tree(blocks).root.children
            return tree.root.html()
    return blocks + body  # legacy page with no generated participant header


def verify_transcript(value, index, transcript_url):
    """Round-trip every original source row; IDs alone are not proof of content."""
    identifier = 'transcri-transcript-' + index['source_sha256'][:32]
    root = Tree(value).root
    matches = [n for n in root.children if isinstance(n, Node) and n.tag == 'details' and n.attrs.get('data-id') == identifier]
    if len(matches) != 1 or 'open' in matches[0].attrs:
        return False
    def rows(node):
        result = []
        for child in node.children:
            if isinstance(child, Node):
                if child.tag == 'p':
                    result.append((child.text(), [n.attrs.get('href') for n in child.children if isinstance(n, Node) and n.tag == 'a']))
                else:
                    result.extend(rows(child))
        return result
    expected = rows(Tree(transcript_block(index, transcript_url)).root)
    actual = rows(matches[0])
    return actual == expected


def verify_layout(value, index, transcript_url, media_marker, assets):
    if not verify_transcript(value, index, transcript_url):
        return False
    nodes = [n for n in Tree(value).root.children if isinstance(n, Node)]
    previews = [a['asset_id'] for a in assets if a['role'] == 'preview']
    originals = {a['asset_id'] for a in assets if a['role'] == 'original'}
    media = [i for i, n in enumerate(nodes) if n.tag == 'attachment-component' and n.attrs.get('src') in previews]
    timecodes = [i for i, n in enumerate(nodes) if n.attrs.get('data-id') == media_marker]
    transcripts = [i for i, n in enumerate(nodes) if n.attrs.get('data-id') == 'transcri-transcript-' + index['source_sha256'][:32]]
    if len(media) != 1 or len(timecodes) != 1 or len(transcripts) != 1 or not nodes:
        return False
    if any(n.tag == 'attachment-component' and n.attrs.get('src') in originals for n in nodes):
        return False
    return nodes[0].tag == 'p' and nodes[0].text().strip().startswith(PARTICIPANTS) and 0 < media[0] < timecodes[0] < transcripts[0]
