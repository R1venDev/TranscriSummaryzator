'use strict';

(() => {
  const style = document.createElement('style');
  style.textContent = '.plane-action{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:12px 0;font:14px/1.5 system-ui,sans-serif}.plane-action button{border:1px solid #597797;border-radius:8px;padding:8px 12px;background:#203c5c;color:#edf5ff;cursor:pointer;font:inherit}.plane-action button:disabled{opacity:.6;cursor:wait}.plane-action a,#plane-meeting-page a{color:#7fb7f5}.plane-action .plane-error{color:#d26370}.plane-action .plane-note{opacity:.8}.plane-action button:focus-visible,.plane-action a:focus-visible{outline:2px solid #7fb7f5;outline-offset:3px}#plane-meeting-page{margin:14px 0;font:14px/1.5 system-ui,sans-serif}';
  document.head.append(style);
  const createdStates = new Set(['created', 'sent', 'succeeded', 'exists', 'linked']);
  const pendingStates = new Set(['pending', 'creating', 'queued', 'processing', 'in_progress', 'sending']);
  const unavailableStates = new Set(['unavailable', 'unconfigured', 'not_configured', 'credential_required', 'blocked']);
  let snapshot = null;
  let loadError = '';
  let requestId = 0;
  let pollTimer = null;
  const sending = new Set();
  const itemErrors = new Map();

  function jobId() {
    return document.body.dataset.planeJobId || new URLSearchParams(location.search).get('id') || '';
  }

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  function link(text, href) {
    try {
      const url = new URL(href, location.href);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null;
      const anchor = node('a', text);
      anchor.href = url.href;
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      return anchor;
    } catch (_) { return null; }
  }

  function settingsLink() {
    const anchor = node('a', 'Настроить Plane');
    anchor.href = '/plane-settings';
    return anchor;
  }

  function button(text, action, disabled = false) {
    const element = node('button', text);
    element.type = 'button';
    element.disabled = disabled;
    if (action) element.addEventListener('click', action);
    return element;
  }

  async function request(path, body) {
    const options = {cache: 'no-store', credentials: 'same-origin'};
    if (body !== undefined) {
      options.method = 'POST';
      options.headers = {'Content-Type': 'application/json', 'X-Requested-With': 'TranscriSummaryzator-Admin'};
      options.body = JSON.stringify(body);
    }
    const response = await fetch('/api/summary/plane/' + path, options);
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error('Не удалось прочитать ответ Plane. Обновите статус.'); }
    if (!response.ok) throw new Error(data.error || 'Операция Plane не выполнена');
    return data;
  }

  function stale() {
    const displayed = document.body.dataset.planeGenerationId;
    return displayed && snapshot && String(displayed) !== String(snapshot.generation_id);
  }

  function renderMeeting() {
    const target = document.querySelector('#plane-meeting-page');
    if (!target) return;
    target.classList.add('plane-action');
    target.replaceChildren();
    if (!snapshot || !snapshot.meeting_page) return;
    const page = snapshot.meeting_page;
    if (stale()) {
      target.textContent = 'Версия конспекта изменилась. Обновите страницу перед созданием Wiki-страницы.';
      return;
    }
    const remote = page.remote_url && link('Wiki-страница встречи в Plane ↗', page.remote_url);
    if (remote) target.append(remote);
    if (page.state === 'update_available') {
      target.append(node('span', 'Конспект изменился. Изменения не отправлены в Wiki, чтобы сохранить ручные правки в Plane.', 'plane-note'));
    } else if (!remote && createdStates.has(page.state)) {
      target.textContent = 'Wiki-страница встречи создана в Plane.';
    } else if (sending.has('page:meeting') || pendingStates.has(page.state)) {
      target.append(node('span', 'Создаётся Wiki-страница встречи в Plane…', 'plane-note'));
    } else if (page.state === 'submission_unknown') {
      target.append(node('span', 'Ожидается подтверждение создания Wiki-страницы встречи в Plane.', 'plane-note'), button('Обновить статус', load));
    } else if (unavailableStates.has(page.state)) {
      target.append(node('span', page.error || 'Для Wiki-страницы требуется подключение Plane.', 'plane-note'), settingsLink());
    } else if (!remote && ['not_created', 'error', 'failed'].includes(page.state)) {
      target.append(button('Создать страницу встречи', () => create({kind: 'page', item_id: 'meeting'})));
    }
    const error = itemErrors.get('page:meeting') || page.error;
    if (error && !unavailableStates.has(page.state)) target.append(node('span', error, 'plane-error'));
  }

  function render() {
    renderMeeting();
    document.querySelectorAll('.plane-action[data-plane-kind][data-plane-item-id]').forEach(target => {
      target.replaceChildren();
      target.setAttribute('aria-live', 'polite');
      const kind = target.dataset.planeKind;
      const id = target.dataset.planeItemId;
      const key = kind + ':' + id;
      if (!['task', 'hypothesis'].includes(kind) || !id) return;
      if (!snapshot) {
        target.append(node('span', loadError || 'Загружаю статус Plane…', loadError ? 'plane-error' : 'plane-note'));
        if (loadError) target.append(button('Обновить статус', load), settingsLink());
        return;
      }
      if (stale()) {
        target.append(node('span', 'Версия конспекта изменилась. Обновите страницу перед отправкой в Plane.', 'plane-note'));
        return;
      }
      const item = (snapshot.items || []).find(value => value.kind === kind && String(value.item_id) === id);
      if (!item || !snapshot.generation_id) {
        target.append(node('span', 'Этот пункт недоступен для отправки в Plane. Обновите страницу.', 'plane-note'));
        return;
      }
      const remote = item.remote_url && link('Открыть в Plane ↗', item.remote_url);
      if (remote || createdStates.has(item.state) || item.state === 'update_available') {
        target.append(node('span', kind === 'hypothesis' ? 'Предложение отправлено в Plane.' : 'Задача создана в Plane.', 'plane-note'));
        if (remote) target.append(remote);
        if (item.state === 'update_available') target.append(node('span', 'Конспект изменился. Изменения не отправлены, чтобы сохранить ручные правки в Plane.', 'plane-note'));
        return;
      }
      if (item.state === 'identity_ambiguous') {
        target.append(node('span', item.error || 'Не удалось однозначно сопоставить этот пункт с существующим объектом Plane. Отправка заблокирована.', 'plane-note'));
        return;
      }
      if (unavailableStates.has(item.state)) {
        target.append(node('span', item.error || 'Для отправки требуется подключение Plane.', 'plane-note'), settingsLink());
        return;
      }
      if (item.state === 'submission_unknown') {
        target.append(node('span', 'Plane пока не подтвердил результат отправки. Повторное создание временно недоступно.', 'plane-note'), button('Обновить статус', load));
        if (item.error) target.append(node('span', item.error, 'plane-error'));
        return;
      }
      const pending = sending.has(key) || pendingStates.has(item.state);
      const dirty = target.dataset.planeDirty === 'true';
      target.append(button(pending ? 'Отправляется в Plane…' : kind === 'hypothesis' ? 'Отправить гипотезу в Plane' : 'Создать задачу в Plane',
        () => create(item), pending || dirty));
      if (dirty) target.append(node('span', 'Сначала сохраните изменения карточки.', 'plane-note'));
      else if (kind === 'hypothesis') target.append(node('span', 'Будет создано предложение для проверки.', 'plane-note'));
      const error = itemErrors.get(key) || item.error || loadError;
      if (error) target.append(node('span', error, 'plane-error'));
    });
  }

  async function create(item) {
    const key = item.kind + ':' + item.item_id;
    if (sending.has(key) || stale() || !snapshot || !snapshot.generation_id) return;
    sending.add(key); itemErrors.delete(key); render();
    try {
      await request('create', {job_id: Number(jobId()), kind: item.kind, item_id: item.item_id, generation_id: snapshot.generation_id});
      await load();
    } catch (error) {
      itemErrors.set(key, error.message);
      await load();
    } finally { sending.delete(key); render(); }
  }

  async function load() {
    const id = jobId();
    if (!/^\d+$/.test(id)) return;
    const sequence = ++requestId;
    clearTimeout(pollTimer);
    try {
      const data = await request('items?job_id=' + encodeURIComponent(id));
      if (sequence !== requestId) return;
      snapshot = data;
      loadError = '';
      render();
      if ((data.items || []).some(item => pendingStates.has(item.state)) || (data.meeting_page && pendingStates.has(data.meeting_page.state))) {
        pollTimer = setTimeout(load, 4000);
      }
    } catch (error) {
      if (sequence !== requestId) return;
      loadError = error.message;
      render();
    }
  }

  document.addEventListener('plane:refresh', load);
  document.addEventListener('plane:render', render);
  window.addEventListener('pagehide', () => { clearTimeout(pollTimer); });
  load();
})();
