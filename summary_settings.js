'use strict';
const writerKeysNode = document.querySelector('#writerKeys');
const judgeKeysNode = document.querySelector('#judgeKeys');
const messageNode = document.querySelector('#message');
let snapshot = {keys: []};

function message(value, bad = false) {
  messageNode.textContent = String(value || '');
  messageNode.className = bad ? 'error' : 'ok';
}

async function load() {
  try {
    const response = await fetch('/api/summary/credentials', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось загрузить ключи');
    snapshot = data;
    render();
  } catch (error) {
    writerKeysNode.textContent = error.message;
    judgeKeysNode.textContent = error.message;
    message(error.message, true);
  }
}

async function post(action, body) {
  const response = await fetch('/api/summary/credentials/' + action, {
    method: 'POST', credentials: 'same-origin',
    headers: {'Content-Type': 'application/json', 'X-Requested-With': 'TranscriSummaryzator-Admin'},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Операция не выполнена');
  await load();
  return data.result;
}

function button(label, action, kind = 'secondary') {
  const node = document.createElement('button');
  node.type = 'button'; node.className = kind; node.textContent = label;
  node.addEventListener('click', async () => {
    node.disabled = true; message('');
    try { await action(); message('Сохранено'); }
    catch (error) { message(error.message, true); }
    finally { node.disabled = false; }
  });
  return node;
}

function renderRole(role, keysNode) {
  const keys = snapshot.keys.filter(key => key.role === role);
  keysNode.replaceChildren();
  if (!keys.length) { keysNode.textContent = 'Ключей пока нет'; return; }
  keys.forEach((key, index) => {
    const item = document.createElement('div'); item.className = 'key';
    const head = document.createElement('div'); head.className = 'key-head';
    const name = document.createElement('span'); name.className = 'key-name'; name.textContent = key.label;
    const mask = document.createElement('span'); mask.className = 'pill'; mask.textContent = key.mask;
    const state = document.createElement('span'); state.className = 'pill'; state.textContent = key.enabled ? key.status : 'отключён';
    head.append(name, mask, state);
    if (key.primary) { const primary = document.createElement('span'); primary.className = 'pill'; primary.textContent = 'Основной'; head.append(primary); }
    const meta = document.createElement('div'); meta.className = 'meta';
    const checked = key.checked_at || 'не проверен';
    const remaining = key.limit_remaining == null ? 'неизвестен' : String(key.limit_remaining);
    meta.textContent = `Версия ${key.version} · Проверен: ${checked} · Доступный лимит ключа: ${remaining} · Workspace: ${key.workspace_id || 'неизвестен'}. Проверка не подтверждает модель, Batch или политику доступа.`;
    const controls = document.createElement('div'); controls.className = 'controls';
    controls.append(button('Проверить', () => post('check', {id: key.id})));
    if (!key.primary) controls.append(button('Сделать основным', () => post('order', {role, ids: [key.id, ...keys.filter(x => x.id !== key.id).map(x => x.id)]})));
    if (index > 0) controls.append(button('Выше', () => {
      const ids = keys.map(x => x.id); [ids[index-1], ids[index]] = [ids[index], ids[index-1]]; return post('order', {role, ids});
    }));
    if (index < keys.length - 1) controls.append(button('Ниже', () => {
      const ids = keys.map(x => x.id); [ids[index+1], ids[index]] = [ids[index], ids[index+1]]; return post('order', {role, ids});
    }));
    controls.append(button(key.enabled ? 'Отключить новые отправки' : 'Включить', () => post('enabled', {id: key.id, enabled: !key.enabled})));
    controls.append(button('Удалить локально', async () => {
      if (window.confirm('Удалить сохранённый секрет? Это не отзывает ключ у провайдера; активные внешние запросы блокируют удаление.')) await post('delete', {id: key.id});
    }, 'danger'));
    const form = document.createElement('form'); form.className = 'replace'; form.autocomplete = 'off';
    const label = document.createElement('label'); label.textContent = 'Заменить секрет';
    const input = document.createElement('input'); input.type = 'password'; input.minLength = 16; input.required = true; input.autocomplete = 'new-password'; label.append(input);
    const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Заменить';
    form.append(label, submit);
    form.addEventListener('submit', async event => {
      event.preventDefault(); submit.disabled = true; message('');
      const value = input.value; input.value = '';
      try { await post('replace', {id: key.id, key: value}); message('Ключ заменён. Проверьте новую версию.'); }
      catch (error) { message(error.message, true); }
      finally { submit.disabled = false; }
    });
    item.append(head, meta, controls, form);
    keysNode.append(item);
  });
}

function render() {
  renderRole('writer', writerKeysNode);
  renderRole('judge', judgeKeysNode);
}

for (const [role, formId] of [['writer', 'writerAddForm'], ['judge', 'judgeAddForm']]) {
  document.querySelector('#' + formId).addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget; const submit = form.querySelector('button');
    const label = form.elements.label.value.trim(); const key = form.elements.key.value;
    form.elements.key.value = ''; submit.disabled = true; message('');
    try { await post('add', {role, label, key}); form.elements.label.value = ''; message('Ключ сохранён; выполните бесплатную проверку'); }
    catch (error) { message(error.message, true); }
    finally { submit.disabled = false; }
  });
}
load();
