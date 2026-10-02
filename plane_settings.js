'use strict';

(() => {
  const form = document.querySelector('#planeForm');
  const fields = document.querySelector('#planeFields');
  const connection = document.querySelector('#connectionStatus');
  const message = document.querySelector('#message');
  const reload = document.querySelector('#reloadPlane');
  const check = document.querySelector('#checkPlane');
  const textFields = ['base_url', 'workspace_slug', 'project_id', 'collection_id', 'parent_page_id', 'task_type_id', 'hypothesis_type_id'];
  const switches = ['auto_tasks', 'auto_hypotheses', 'auto_meeting_page'];
  let settings = null;
  let busy = false;

  function notify(value, bad = false) {
    message.textContent = String(value || '');
    message.className = 'status ' + (bad ? 'error' : 'success');
  }

  function setBusy(value) {
    busy = value;
    fields.disabled = value || !settings;
    check.disabled = value || !settings;
    reload.disabled = value;
    form.setAttribute('aria-busy', String(value));
  }

  async function request(path, body) {
    const options = {credentials: 'same-origin', cache: 'no-store'};
    if (body !== undefined) {
      options.method = 'POST';
      options.headers = {'Content-Type': 'application/json', 'X-Requested-With': 'TranscriSummaryzator-Admin'};
      options.body = JSON.stringify(body);
    }
    const response = await fetch('/api/summary/plane/' + path, options);
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error('Сервер вернул ответ, который не удалось прочитать. Обновите страницу.'); }
    if (!response.ok) {
      const error = new Error(data.error || 'Операция не выполнена');
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function showConnection(status = {}) {
    const state = String(status.state || '');
    const failed = ['error', 'failed', 'unavailable', 'invalid', 'unauthorized'].includes(state);
    const ready = ['ok', 'ready', 'connected', 'verified'].includes(state);
    connection.textContent = String(status.message || (settings && settings.configured
      ? 'Подключение настроено. Проверьте доступ к Plane.' : 'Добавьте параметры подключения и API-ключ.'));
    connection.className = 'status connection' + (failed ? ' error' : ready ? ' success' : '');
  }

  function showOptions(selector, values) {
    const list = document.querySelector(selector);
    list.replaceChildren();
    if (!Array.isArray(values)) return;
    values.forEach(value => {
      if (!value || value.id == null) return;
      const option = document.createElement('option');
      option.value = String(value.id);
      option.textContent = String(value.name || value.title || value.id);
      list.append(option);
    });
  }

  function apply(data) {
    if (!data.settings || data.settings.revision == null) throw new Error('Ответ сервера не содержит версию настроек.');
    settings = data.settings;
    showTypes(data.options || {}, settings);
    textFields.forEach(name => { form.elements[name].value = String(settings[name] || ''); });
    switches.forEach(name => {
      form.elements[name].checked = name === 'auto_meeting_page' ? settings[name] !== false : settings[name] === true;
    });
    form.elements.api_key.value = '';
    document.querySelector('#keyMask').textContent = settings.key_mask ? 'Сохранённый ключ: ' + settings.key_mask : 'Сохранённого ключа нет';
    showConnection(data.status);
    showOptions('#planeProjects', data.options && data.options.projects);
    showOptions('#planeCollections', data.options && data.options.collections);
    reload.hidden = true;
  }

  function showTypes(options, values) {
    ['task_type_id', 'hypothesis_type_id'].forEach(name => {
      const list = form.elements[name];
      const selected = String(values[name] || '');
      list.replaceChildren(new Option('Тип проекта по умолчанию', ''));
      if (options.types_project_id === values.project_id && Array.isArray(options.types)) {
        options.types.filter(t => t.is_active === true && t.is_epic === false).forEach(t => {
          list.append(new Option(String(t.name), String(t.id)));
        });
      }
      if (selected && !Array.from(list.options).some(t => t.value === selected)) {
        list.append(new Option('Выбранный тип — требуется проверка проекта', selected));
      }
      list.value = selected;
    });
  }

  form.elements.project_id.addEventListener('change', () => {
    showTypes({}, {project_id: form.elements.project_id.value});
  });

  async function load() {
    if (busy) return;
    setBusy(true);
    try { apply(await request('settings')); }
    catch (error) { notify(error.message, true); reload.hidden = false; }
    finally { setBusy(false); }
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !settings) return;
    const body = {expected_revision: settings.revision};
    textFields.forEach(name => { body[name] = form.elements[name].value.trim(); });
    switches.forEach(name => { body[name] = form.elements[name].checked; });
    if (form.elements.api_key.value) body.api_key = form.elements.api_key.value;
    form.elements.api_key.value = '';
    setBusy(true); notify('Сохраняю…');
    try {
      const data = await request('settings', body);
      apply(data.settings ? data : await request('settings'));
      notify('Настройки сохранены.');
    } catch (error) {
      notify(error.message + (error.status === 409 ? ' Загрузите текущие настройки и повторите изменения.' : ''), true);
      if (error.status === 409) reload.hidden = false;
    } finally { setBusy(false); }
  });

  check.addEventListener('click', async () => {
    if (busy || !settings) return;
    setBusy(true); notify('Проверяю сохранённое подключение…');
    try {
      const result = await request('check', {});
      // Refresh option lists without overwriting edits still present in the form.
      const current = await request('settings');
      showOptions('#planeProjects', current.options && current.options.projects);
      showOptions('#planeCollections', current.options && current.options.collections);
      showTypes(current.options || {}, {project_id: form.elements.project_id.value,
        task_type_id: form.elements.task_type_id.value, hypothesis_type_id: form.elements.hypothesis_type_id.value});
      const status = result.status || current.status || result;
      showConnection(status);
      notify(status.message || 'Проверка завершена.', ['error', 'failed', 'unavailable', 'invalid', 'unauthorized'].includes(status.state));
    } catch (error) { notify(error.message, true); }
    finally { setBusy(false); }
  });

  reload.addEventListener('click', () => { notify(''); load(); });
  load();
})();
