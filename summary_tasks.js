'use strict';

const params = new URLSearchParams(location.search);
const jobId = params.get('id') || '';
const notice = document.querySelector('#notice');
const cards = document.querySelector('#cards');
const summaryLink = document.querySelector('#summaryLink');
const resultLink = document.querySelector('#resultLink');
const editable = ['title', 'description', 'assignee', 'due', 'priority', 'recipient', 'discussion_status'];
let selected = null;

function show(text, bad = false) {
  notice.textContent = text;
  notice.classList.toggle('error', bad);
}

function make(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value != null) element.textContent = String(value);
  return element;
}

function field(form, task, name, caption, options = null) {
  const wrapper = make('label', name === 'title' || name === 'description' ? 'wide' : '', caption);
  let input;
  if (options) {
    input = make('select');
    options.forEach(([value, label]) => {
      const option = make('option', '', label);
      option.value = value;
      input.append(option);
    });
  } else {
    input = make(name === 'description' ? 'textarea' : 'input');
    if (name !== 'description') input.type = 'text';
  }
  input.name = name;
  input.value = task[name] == null ? '' : task[name];
  if (name === 'title' || name === 'description') input.required = true;
  if (name === 'title') input.maxLength = 300;
  if (name === 'description') input.maxLength = 20000;
  if (name !== 'title' && name !== 'description' && name !== 'discussion_status') input.maxLength = 1000;
  wrapper.append(input); form.append(wrapper);
}

function references(task) {
  const sources = selected.source_refs[task.action_id] || [];
  const detail = make('details', 'source');
  detail.append(make('summary', '', `Источники (${sources.length})`));
  sources.forEach(ref => {
    const paragraph = make('p');
    const link = make('a', '', `${ref.source_id} · ${ref.timecode}`);
    link.href = `/result?id=${encodeURIComponent(jobId)}#t-${Number(ref.start_ms)}`;
    paragraph.append(link, document.createTextNode(` · ${ref.speaker}`));
    detail.append(paragraph, make('p', 'quote', ref.text));
  });
  return detail;
}

function pendingChanges(form, task) {
  const changes = {};
  editable.forEach(name => {
    const input = form.elements[name];
    let value = input.value.trim();
    if (!['title', 'description', 'discussion_status'].includes(name) && !value) value = null;
    if (value !== task[name]) changes[name] = value;
  });
  return changes;
}

async function save(form, task, message, button) {
  const changes = pendingChanges(form, task);
  if (!Object.keys(changes).length) {
    message.textContent = 'Изменений нет'; return;
  }
  button.disabled = true; message.className = 'status'; message.textContent = 'Сохраняю…';
  try {
    const response = await fetch(`/api/summary/tasks/${encodeURIComponent(task.action_id)}?id=${encodeURIComponent(jobId)}`, {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'Content-Type': 'application/json', 'X-Requested-With': 'TranscriSummaryzator-Admin'},
      body: JSON.stringify({expected_generation_id: selected.generation_id, expected_revision: task.revision, changes}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Изменение не сохранено');
    selected = data;
    render();
    show('Карточка сохранена. Конспект и экспорты используют новую ревизию.');
  } catch (error) {
    message.className = 'status error';
    message.textContent = error.message + ' При конфликте обновите страницу и сравните текущую версию.';
  } finally {
    button.disabled = false;
  }
}

function render() {
  cards.replaceChildren();
  if (!selected.tasks.length) {
    cards.append(make('p', 'muted', 'В выбранной версии конспекта нет карточек задач.'));
    return;
  }
  selected.tasks.forEach((task, index) => {
    const article = make('article', 'card');
    article.append(make('h2', '', `T-${String(index + 1).padStart(2, '0')}. ${task.title}`));
    const metadata = make('p', 'muted', `ID ${task.action_id} · ревизия ${task.revision} · исполнитель: ${task.assignee || 'Не назначен'}`);
    article.append(metadata);
    const form = make('form', 'grid');
    field(form, task, 'title', 'Название');
    field(form, task, 'description', 'Описание');
    field(form, task, 'assignee', 'Исполнитель (необязательно)');
    field(form, task, 'due', 'Срок (необязательно)');
    field(form, task, 'priority', 'Приоритет (необязательно)');
    field(form, task, 'recipient', 'Получатель (необязательно)');
    field(form, task, 'discussion_status', 'Статус по обсуждению', [
      ['proposed', 'Предложено / нужно распределить'],
      ['committed', 'Принято в обсуждении'],
      ['in_progress', 'В работе по словам участника'],
      ['unknown', 'Не установлен'],
    ]);
    article.append(form, references(task));
    const actions = make('div', 'actions');
    const button = make('button', 'primary', 'Сохранить карточку');
    button.type = 'submit';
    const message = make('span', 'status');
    actions.append(button, message); form.append(actions);
    form.addEventListener('submit', event => { event.preventDefault(); save(form, task, message, button); });
    cards.append(article);
  });
}

async function load() {
  if (!/^\d+$/.test(jobId)) { show('Неверный номер записи', true); return; }
  summaryLink.href = `/summary?id=${encodeURIComponent(jobId)}`;
  resultLink.href = `/result?id=${encodeURIComponent(jobId)}`;
  try {
    const response = await fetch(`/api/summary/tasks?id=${encodeURIComponent(jobId)}`, {cache: 'no-store', credentials: 'same-origin'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Карточки недоступны');
    selected = data;
    show(`Версия конспекта ${selected.generation_id}. Карточки сохраняются локально.`);
    render();
  } catch (error) {
    show(error.message, true);
  }
}

load();
