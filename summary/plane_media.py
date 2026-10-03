"""Native Plane video attachments and timecodes, from accepted local evidence.

Uses the existing Plane outbox database and scheduler. No model or speech calls.
Presigned credentials are encrypted, never returned by the public API.
"""
from __future__ import annotations

import fcntl
import hashlib
import html
import http.client
import json
import mimetypes
import os
from pathlib import Path
import ssl
import subprocess
import time
import urllib.parse
import uuid

from scripts.summary_credentials import _fernet, load_master_key

VIDEO_SUFFIXES = {'.mp4', '.mkv', '.mov', '.webm', '.m4v', '.avi'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def media_spec(job, output, view, source_index, data_root):
    """An imported transcript needs an explicit verified media binding.

    Ordinary uploads use the queue's media path/hash. Never guess by filename.
    Only files under the configured data root can leave the application.
    """
    source = Path(job['source_path'])
    expected = job['content_sha256']
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        binding = Path(output) / 'plane_media_source.json'
        if not binding.is_file():
            return None
        info = json.loads(binding.read_text())
        if info.get('source_sha256') != view.source_sha256:
            raise ValueError('Источник видео не соответствует конспекту')
        source, expected = Path(info['path']), info['media_sha256']
    source = source.resolve(strict=True)
    source.relative_to(Path(data_root).resolve(strict=True))
    if source.suffix.lower() not in VIDEO_SUFFIXES or not source.is_file():
        raise ValueError('Исходное видео недоступно')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Нет подтверждённой identity видео')
    chapters = []
    for chapter in view.rendered['summary.json']['timecodes']:
        start = source_index['by_id'][chapter['start_id']]['start_ms']
        end = source_index['by_id'][chapter['end_id']]['end_ms']
        if start < 0 or end < start:
            raise ValueError('Неверный диапазон таймкода')
        chapters.append({'seconds': int(start // 1000), 'label': str(chapter['topic']),
                         'start_id': chapter['start_id'], 'end_id': chapter['end_id']})
    chapters.sort(key=lambda c: c['seconds'])
    return {'path': str(source), 'sha256': expected, 'name': Path(job['original_name']).name,
            'source_sha256': view.source_sha256, 'chapters': chapters}


def clock(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}' if hours else f'{minutes:02}:{seconds:02}'


def native_blocks(identity, assets, chapters):
    """App-issued editor nodes only; untrusted labels are escaped."""
    esc = lambda x: html.escape(str(x), quote=True)
    result = []
    for asset in assets:
        if asset['role'] != 'preview':
            continue
        block = str(uuid.uuid5(uuid.NAMESPACE_URL, identity + ':' + asset['role']))
        identifier = str(uuid.UUID(asset['asset_id']))
        preview = asset['role'] == 'preview'
        result.append(f'<attachment-component id="{block}" data-id="{block}" src="{identifier}" '
                      f'data-name="{esc(asset["name"])}" data-file-type="{esc(asset["type"])}" '
                      f'data-file-size="{int(asset["size"])}" data-preview="{str(preview).lower()}" '
                      f'data-accepted-file-type="{"video" if preview else "all"}" status="uploaded" '
                      'data-spacing-group="container"></attachment-component>')
    marker = 'plane-timecodes-' + identity
    lines = '<br>'.join(f'<span>{clock(c["seconds"])} {esc(c["label"])}</span>' for c in chapters)
    result.append(f'<div data-id="{marker}" data-block-type="callout-component" '
                  'data-icon-name="Clock" data-icon-color="#7dd3fc" data-logo-in-use="icon" '
                  f'data-background="" data-spacing-group="container"><p>⏱ Таймкоды</p><p>{lines}</p></div>')
    return ''.join(result)


def upload_file(client, upload_data, path, mime):
    """Stream multipart bytes without the API key, redirects or huge RAM buffers."""
    from summary.plane import PlaneError
    url = urllib.parse.urlsplit(upload_data['url'])
    base = urllib.parse.urlsplit(client.base)
    # Some self-hosted Plane installations return a Docker-only MinIO origin.
    # A verified reverse proxy may be mapped explicitly; preserve bucket/path
    # and all signed form fields. Never infer a replacement from model data.
    mapping = json.loads(os.environ.get('TRANSCRI_PLANE_STORAGE_ORIGIN_MAP', '{}'))
    origin = url.scheme + '://' + url.netloc
    if origin in mapping:
        target = urllib.parse.urlsplit(mapping[origin])
        if target.scheme != 'https' or target.path not in ('', '/') or target.query or target.fragment or target.username or target.password:
            raise PlaneError('Неверный HTTPS proxy хранилища Plane')
        url = url._replace(scheme=target.scheme, netloc=target.netloc)
    # Extra storage origins must be explicitly configured by the operator.
    trusted = {(base.scheme, base.netloc)}
    for origin in os.environ.get('TRANSCRI_PLANE_STORAGE_ORIGINS', '').split(','):
        if origin:
            parsed = urllib.parse.urlsplit(origin.strip())
            if parsed.scheme != 'https' or parsed.path not in ('', '/') or parsed.username or parsed.password:
                raise PlaneError('Неверный разрешённый адрес хранилища Plane')
            trusted.add((parsed.scheme, parsed.netloc))
    if (url.scheme, url.netloc) not in trusted or url.username or url.password or url.fragment:
        raise PlaneError('Адрес загрузки не входит в разрешённые хранилища Plane')
    boundary = 'transcri-' + uuid.uuid4().hex
    prefix = bytearray()
    for key, value in upload_data['fields'].items():
        if not isinstance(key, str) or any(c in key for c in '\r\n"'):
            raise PlaneError('Некорректный контракт загрузки Plane')
        prefix.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    # ASCII transport filename; original Unicode name is stored by Plane API.
    prefix.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="video{Path(path).suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    suffix = f'\r\n--{boundary}--\r\n'.encode()
    connection = (http.client.HTTPSConnection(url.hostname, url.port, timeout=120, context=ssl.create_default_context())
                  if url.scheme == 'https' else http.client.HTTPConnection(url.hostname, url.port, timeout=120))
    try:
        connection.putrequest('POST', url.path + ('?' + url.query if url.query else ''))
        connection.putheader('Content-Type', 'multipart/form-data; boundary=' + boundary)
        connection.putheader('Content-Length', str(len(prefix) + Path(path).stat().st_size + len(suffix)))
        connection.endheaders()
        connection.send(prefix)
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        response.read(65536)
        if response.status not in (200, 201, 204):
            raise PlaneError(f'Хранилище Plane HTTP {response.status}; видео ещё не опубликовано')
    except (OSError, http.client.HTTPException):
        raise PlaneError('Загрузка видео прервана; состояние сохранено') from None
    finally:
        connection.close()


def init_table(db):
    db.execute('''CREATE TABLE IF NOT EXISTS page_media(
        delivery_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, spec TEXT NOT NULL,
        state TEXT NOT NULL, assets TEXT NOT NULL DEFAULT '[]', error TEXT NOT NULL DEFAULT '',
        next_attempt REAL NOT NULL DEFAULT 0, updated REAL NOT NULL)''')


def enqueue(store, job_id, generation_id, spec):
    from summary.plane import _json, _delivery_destination
    if not spec:
        return
    with store._guard(), store._db() as db:
        settings = store._settings(store._row(db))
        row = db.execute("SELECT * FROM deliveries WHERE job_id=? AND kind='page' AND destination=?",
                         (str(job_id), _json(_delivery_destination(settings, 'page')))).fetchone()
        if not row:
            return
        meeting = db.execute('SELECT * FROM meetings WHERE job_id=?', (str(job_id),)).fetchone()
        if meeting['generation_id'] != generation_id or meeting['source_id'] != spec['source_sha256']:
            raise ValueError('Саммари изменилось до постановки видео в очередь')
        old = db.execute('SELECT * FROM page_media WHERE delivery_id=?', (row['id'],)).fetchone()
        if old:
            if old['spec'] == '{}' and not json.loads(old['assets']):
                db.execute("UPDATE page_media SET source_id=?,spec=?,state='queued',error='',next_attempt=0,updated=? WHERE delivery_id=?",
                           (spec['source_sha256'], _json(spec), time.time(), row['id']))
                return
            # Already published native blocks belong to their original generation;
            # never replace them or human edits silently on regeneration.
            if old['spec'] == _json(spec) or old['state'] == 'published':
                return
            # Sent/allocated assets remain bound to an immutable spec. A new
            # source/chapter plan cannot relabel old assets or race a worker.
            return
        else:
            db.execute("INSERT INTO page_media(delivery_id,source_id,spec,state,updated) VALUES(?,?,?,'queued',?)",
                       (row['id'], spec['source_sha256'], _json(spec), time.time()))
            store._event(db, row['id'], 'media_enqueued', {'source_id': spec['source_sha256']})


def unavailable(store, job_id, source_id):
    """Optional media errors never break task/page controls or generation."""
    from summary.plane import _json, _delivery_destination
    with store._guard(), store._db() as db:
        settings = store._settings(store._row(db))
        row = db.execute("SELECT id FROM deliveries WHERE job_id=? AND kind='page' AND destination=?",
                         (str(job_id), _json(_delivery_destination(settings, 'page')))).fetchone()
        if row:
            db.execute("INSERT OR IGNORE INTO page_media(delivery_id,source_id,spec,state,error,updated) VALUES(?,?,?,'blocked',?,?)",
                       (row['id'], source_id, '{}', 'Исходное видео недоступно или не подтверждено. Конспект и карточки доступны.', time.time()))


def prepare(spec, cache):
    """Copy-free original and a lossless container conversion for H.264/AAC MKV."""
    from summary.plane import PlaneError
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise PlaneError('Исходное видео изменилось; загрузка заблокирована')
    try:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'stream=codec_name,codec_type', '-of', 'json', str(path)], stderr=subprocess.DEVNULL))
    except (OSError, subprocess.SubprocessError, ValueError):
        raise PlaneError('Не удалось прочитать формат исходного видео') from None
    codecs = {s['codec_name'] for s in probe['streams'] if s['codec_type'] in ('video', 'audio')}
    if 'h264' not in codecs or not codecs.issubset({'h264', 'aac', 'mp3'}):
        raise PlaneError('Видео требует перекодирования для браузера; оригинал сохранён, автоматическое перекодирование не включено')
    original = {'role': 'original', 'path': str(path), 'sha256': spec['sha256'], 'name': spec['name'],
                'type': mimetypes.guess_type(path.name)[0] or 'application/octet-stream', 'size': path.stat().st_size}
    if path.suffix.lower() in ('.mp4', '.m4v'):
        return [{**original, 'role': 'preview', 'type': 'video/mp4'}]
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = cache / (spec['sha256'] + '.mp4')
    if not target.is_file():
        temporary = target.with_suffix('.partial.mp4')
        try:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(path), '-map', '0:v:0',
                '-map', '0:a:0?', '-c', 'copy', '-movflags', '+faststart', str(temporary)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except (OSError, subprocess.SubprocessError):
            raise PlaneError('Не удалось подготовить MP4 без перекодирования') from None
    return [{'role': 'preview', 'path': str(target), 'sha256': digest(target),
                      'name': Path(spec['name']).stem + '.mp4', 'type': 'video/mp4', 'size': target.stat().st_size}]


def drain(store):
    """One durable media intent per scheduler tick; separate nonblocking lease.

    No open database transaction or administrative lock during streaming.
    Unknown page PUT is recovered by native block IDs, never by deleting content.
    """
    from summary.plane import PlaneError, _json, _hash, _delivery_destination, EXTERNAL_SOURCE
    fd = os.open(store.path.with_suffix('.media.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        with store._db() as db:
            setting_row = store._row(db)
            settings = store._settings(setting_row)
            row = db.execute("SELECT m.*,d.remote_id,d.external_id,d.destination,d.job_id FROM page_media m JOIN deliveries d ON m.delivery_id=d.id WHERE m.state!='published' AND m.state!='blocked' AND m.next_attempt<=? AND d.destination=? AND d.state IN ('created','update_available') ORDER BY m.updated LIMIT 1", (time.time(), _json(_delivery_destination(settings, 'page')))).fetchone()
            if not row:
                return None
            row = dict(row)
        if row['destination'] != _json(_delivery_destination(settings, 'page')) or not settings['auto_meeting_page']:
            return None
        spec = json.loads(row['spec'])
        assets = json.loads(row['assets'])
        client = store.client_factory(settings, store._key(setting_row))
        cipher = _fernet(store.master_key or load_master_key())
        page_id = str(uuid.UUID(row['remote_id']))
        page_path = client.prefix + 'pages/' + page_id + '/'
        identity = _hash([page_id, spec['sha256']])[:32]

        def save(state='uploading', error='', delay=0):
            with store._db() as db:
                db.execute('UPDATE page_media SET state=?,assets=?,error=?,next_attempt=?,updated=? WHERE delivery_id=?',
                           (state, _json(assets), error, time.time() + delay, time.time(), row['delivery_id']))

        try:
            remote = client.request('GET', page_path)
            body = remote.get('description_html') or ''
            if 'transcrisummaryzator-id:' + row['external_id'] not in body:
                raise PlaneError('Wiki-страница не подтвердила identity встречи')
            if not assets:
                assets = prepare(spec, store.path.parent / 'plane_media')
                save()
            for asset in assets:
                expected_path = Path(spec['path']) if asset['role'] == 'original' or Path(spec['path']).suffix.lower() in ('.mp4', '.m4v') else store.path.parent / 'plane_media' / (spec['sha256'] + '.mp4')
                if asset['role'] not in ('original', 'preview') or Path(asset['path']) != expected_path:
                    raise PlaneError('Identity сохранённого вложения не соответствует исходному видео')
                if expected_path == Path(spec['path']) and asset['sha256'] != spec['sha256']:
                    raise PlaneError('Хеш вложения не соответствует исходному видео')
            for asset in assets:
                # Confirmed page-scoped metadata is the recovery authority.
                if not asset.get('asset_id'):
                    save('allocating')
                    allocation = client.request('POST', client.prefix + 'assets/', {
                        'name': asset['name'], 'type': asset['type'], 'size': asset['size'],
                        'entity_type': 'PAGE_DESCRIPTION', 'entity_identifier': page_id,
                        'external_source': EXTERNAL_SOURCE,
                        'external_id': 'ts-media-' + _hash([page_id, asset['sha256'], asset['role']])[:40]},
                        allowed_statuses=(409,))
                    asset['asset_id'] = str(uuid.UUID(allocation['asset_id']))
                    if allocation.get('upload_data'):
                        asset['upload'] = cipher.encrypt(_json(allocation['upload_data']).encode()).decode()
                    save()
                attachment_path = page_path + 'attachments/' + asset['asset_id'] + '/'
                metadata = client.request('GET', attachment_path)
                if not metadata.get('is_uploaded'):
                    if not asset.get('upload'):
                        raise PlaneError('Plane создал вложение, но адрес загрузки потерян. Нужна сверка администратора; повторное вложение не создаётся')
                    if digest(asset['path']) != asset['sha256']:
                        raise PlaneError('Файл вложения изменился')
                    upload_file(client, json.loads(cipher.decrypt(asset['upload'].encode())), asset['path'], asset['type'])
                    client.request('PATCH', client.prefix + 'assets/' + asset['asset_id'] + '/', {'is_uploaded': True})
                    metadata = client.request('GET', attachment_path)
                    if metadata.get('is_uploaded') is not True:
                        raise PlaneError('Plane не подтвердил сохранение видео')
                asset.pop('upload', None)
                save()
            blocks = native_blocks(identity, assets, spec['chapters'])
            # Re-read immediately before merge. Preserve the complete remote
            # body, including edits since media upload began. API has no CAS.
            body = client.request('GET', page_path).get('description_html') or ''
            marker = 'plane-timecodes-' + identity
            present = [asset['asset_id'] in body for asset in assets if asset['role'] == 'preview']
            if marker in body and all(present):
                save('published')
                return {'state': 'media_published'}
            if marker in body or any(present):
                raise PlaneError('Частично изменены блоки видео; ручные изменения не перезаписаны')
            if 'transcrisummaryzator-id:' + row['external_id'] not in body:
                raise PlaneError('Identity Wiki-страницы изменилась')
            save('publishing')
            # Keep an exact private recovery snapshot before a whole-body PUT.
            # Plane v1 has no conditional update; concurrent editor writes are
            # an API limitation, never an exactly-once/preserve-all guarantee.
            recovery = store.path.parent / 'plane_media' / (identity + '.page-before.json')
            recovery.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with recovery.open('w', encoding='utf-8') as stream:
                os.chmod(recovery, 0o600)
                json.dump({'page_id': page_id, 'source_id': spec['source_sha256'], 'before_html': body,
                           'native_blocks': blocks}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            from summary.plane_wiki import insert_media
            client.request('PUT', page_path, {'description_html': insert_media(body, blocks)})
            after = client.request('GET', page_path).get('description_html') or ''
            if marker not in after or not all(a['asset_id'] in after for a in assets if a['role'] == 'preview'):
                raise PlaneError('Plane не подтвердил нативные блоки видео и таймкодов')
            save('published')
            with store._db() as db:
                store._event(db, row['delivery_id'], 'media_published', {'asset_ids': [a['asset_id'] for a in assets],
                    'chapters': len(spec['chapters']), 'source_id': spec['source_sha256']})
            return {'state': 'media_published'}
        except PlaneError as exc:
            # Bounded retry for transport/server errors, no blind page/file POST.
            retry = 'HTTP 429' in str(exc) or 'HTTP 5' in str(exc) or 'прервана' in str(exc) or 'достоверный ответ' in str(exc)
            save('queued' if retry else 'blocked', str(exc), 300 if retry else 0)
            return {'state': 'media_pending' if retry else 'media_blocked', 'error': str(exc)}
        except Exception:
            save('blocked', 'Не удалось проверить публикацию видео; сохранено состояние для восстановления')
            return {'state': 'media_blocked'}
    finally:
        os.close(fd)


def refresh_layout(store, job_id, source_index, transcript_url):
    """Explicit authorized layout migration of an existing app-owned page.

    Keeps remote summary/manual text and existing uploaded assets. No new page,
    upload, inference or source writes. Exact before/target saved before PUT;
    retry compares its app-issued layout marker, recovering a lost response.
    """
    from summary.plane import PlaneError, _json, _hash, _delivery_destination
    from summary.plane_wiki import Tree, Node, clean_summary, participant_header, transcript_block, resolve_mentions, verify_transcript, verify_layout
    fd = os.open(store.path.with_suffix('.media.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with store._db() as db:
            setting_row = store._row(db)
            settings = store._settings(setting_row)
            row = db.execute("SELECT m.*,d.remote_id,d.external_id,d.destination,d.job_id,d.payload FROM page_media m JOIN deliveries d ON m.delivery_id=d.id WHERE d.job_id=? AND d.destination=? AND d.state IN ('created','update_available')", (str(job_id), _json(_delivery_destination(settings, 'page')))).fetchone()
            if not row:
                raise PlaneError('Нет опубликованного видео этой встречи')
            row = dict(row)
        spec, assets = json.loads(row['spec']), json.loads(row['assets'])
        if spec['source_sha256'] != source_index['source_sha256']:
            raise PlaneError('Источник Wiki не соответствует транскрипции')
        previews = [a for a in assets if a['role'] == 'preview' and a.get('asset_id')]
        if len(previews) != 1:
            raise PlaneError('Нет единственного подтверждённого видео')
        client = store.client_factory(settings, store._key(setting_row))
        page_id = str(uuid.UUID(row['remote_id']))
        path = client.prefix + 'pages/' + page_id + '/'
        identity = _hash([page_id, spec['sha256']])[:32]
        marker = 'transcri-wiki-layout-v2-' + identity
        remote = client.request('GET', path)
        before = remote.get('description_html') or ''
        if 'transcrisummaryzator-id:' + row['external_id'] not in before:
            raise PlaneError('Wiki не подтвердила identity встречи')
        if marker in before:
            if not verify_layout(before, source_index, transcript_url, 'plane-timecodes-' + identity, assets):
                raise PlaneError('Опубликованная транскрипция изменена; автоматическая перезапись отключена')
            return {'state': 'layout_current', 'page_id': page_id}
        metadata = client.request('GET', path + 'attachments/' + str(uuid.UUID(previews[0]['asset_id'])) + '/')
        if metadata.get('is_uploaded') is not True:
            raise PlaneError('Plane не подтвердил видео')
        members = client.list_all(client.prefix + 'members/')
        transcript_id = 'transcri-transcript-' + source_index['source_sha256'][:32]
        generated = Tree(json.loads(row['payload'])['description_html'])
        titles = [n.text() for n in generated.root.children if isinstance(n, Node) and n.tag == 'h1']
        meeting_title = titles[0] if len(titles) == 1 else None
        preserved = clean_summary(before, [a['asset_id'] for a in assets], 'plane-timecodes-' + identity, transcript_id, meeting_title)
        target = (resolve_mentions(participant_header(source_index), members)
                  + native_blocks(identity, previews, spec['chapters'])
                  + transcript_block(source_index, transcript_url) + preserved
                  + '<p>' + marker + '</p>')
        # No CAS in installed Plane API; decline observed concurrent changes.
        if (client.request('GET', path).get('description_html') or '') != before:
            raise PlaneError('Wiki редактируется; повторите обновление после сохранения')
        recovery = store.path.parent / 'plane_media' / (identity + '.layout-v2.json')
        recovery.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with recovery.open('w', encoding='utf-8') as stream:
            os.chmod(recovery, 0o600)
            json.dump({'page_id': page_id, 'source_sha256': spec['source_sha256'],
                       'before_html': before, 'target_html': target}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        with store._db() as db:
            store._event(db, row['delivery_id'], 'wiki_layout_v2_intent', {'source_id': spec['source_sha256'], 'page_id': page_id})
        client.request('PUT', path, {'description_html': target})
        after = client.request('GET', path).get('description_html') or ''
        if marker not in after or previews[0]['asset_id'] not in after or not verify_layout(after, source_index, transcript_url, 'plane-timecodes-' + identity, assets):
            raise PlaneError('Plane не подтвердил новый формат Wiki')
        with store._db() as db:
            store._event(db, row['delivery_id'], 'wiki_layout_v2_saved', {'page_id': page_id, 'turns': len(source_index['by_id']), 'asset_id': previews[0]['asset_id']})
        return {'state': 'layout_updated', 'page_id': page_id, 'turns': len(source_index['by_id'])}
    finally:
        os.close(fd)
