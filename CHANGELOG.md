# Changelog

## 2026-10-03 — видео в Wiki

- Автоматический экспорт исходного видео, MP4 без перекодирования и нативного блока таймкодов в Plane.
- Durable asset intent, encrypted presigned credentials, recovery, readback и видимый media state.
- Без повторного inference, ASR или диаризации.

[Русский README](README.md) · [English README](README.en.md) · [Статус проверки / Validation status](docs/STATUS.md)

## 2026.10.03

- RU: Добавлена детерминированная нормализация repair: точные ID/field coordinates, обновления карточек по полям и атомарное объединение совместимых правок. Некорректная группа не отменяет независимые исправления. Сохранённую попытку с отклонённым repair можно продолжить с S6 без нового writer, сброса бюджета или изменения старой generation. Применяются только проверенные группы; raw, provenance и пользовательские правки сохраняются.
- EN: Repair envelopes now normalize exact IDs/coordinates, whole-task updates and compatible shared-target edits into atomic groups. Invalid groups do not discard independent fixes. A saved repair-validation failure can resume at S6 without another writer, budget reset or mutation of old generations. Only verified groups are applied; native responses, provenance and human edits are retained.

- RU: Подтверждённый исчезнувший Batch больше не удерживает обработку встречи в бесконечном ожидании. Сохраняются неизвестный расход и item-слоты; повторная отправка запрещена, пригодный документ проходит оставшиеся стадии и публикуется с точным неполным review-state. Добавлены приватные create receipts и восстановление после рестарта; gateway cleanup отложен до окончания workflow.
- EN: Confirmed missing Batch records no longer stall a meeting indefinitely. Unknown charges and item slots remain held; missing requests are never reposted. Usable drafts continue through the remaining stages with explicit incomplete-review status. Added private create receipts and restart-safe missing-record state; gateway cleanup waits for workflow completion.
- RU: Убраны фиксированные output caps нового Luna Batch маршрута: резерв рассчитывается по endpoint, фактическому input, оставшимся стадиям и разрешённым деньгам. Частичные отчёты сохраняют пригодные замечания; repair получает таблицу цитат; проверка не отменяет независимые одобренные patches из-за неполноты соседнего bundle. UI показывает число применённых пакетов отдельно от review-state. Речевой pipeline не изменён. Offline replay: 12 пригодных замечаний из сохранённых отчётов 411 реплик; это не live quality PASS.
- EN: Replaced fixed Luna stage output caps with endpoint/context/budget allocation. Grounded findings survive incomplete reports; repair receives source evidence; independent verified patch groups survive unrelated incomplete checks. Applied bundle count is separate from review completeness. Speech processing is unchanged. Offline replay recovered 12 usable findings from a saved 411-turn meeting; live content quality remains to be measured.

## 2026.10.02

### Русский

- Добавлена интеграция Plane: защищённые настройки, Wiki-страница встречи, ручная отправка задач и гипотез, отдельные переключатели автоматизации.
- Отправки сохраняются в локальном журнале. Неизвестный результат POST требует сверки; повторная генерация сохраняет содержимое Plane и ручные правки.
- Добавлены Docker-профили полного речевого контура и обработки готовых стенограмм, отдельные постоянные тома данных и секретов, установщик и обновление образов.
- Зафиксированы публичные зависимости речевых окружений и изменения совместимости DiariZen. Добавлены проверки сборки, запуска лёгкого профиля и сохранения данных после перезапуска.
- Обновлены русский и английский README, схема продукта, инструкции установки и описание ограничений.

**При обновлении:** сохраните резервную копию обоих томов. Существующие конспекты не отправляются в Plane автоматически задним числом. Сохраняйте `plane.sqlite3` вместе с master key для восстановления отправок. Обновление сохраняет паузу обработки; явный флаг `--enable-speech` включает обработку в существующей установке. Для включения собственных изменений в образ используйте `--build`. Подробности — в [инструкции Docker](docs/DOCKER.md) и [руководстве Plane](docs/PLANE.md).

Полное речевое окружение прошло сборку и проверки импорта библиотек. Скачивание весов, обработка аудио на GPU и качество конспектов требуют отдельной проверки. Дата раздела обозначает набор изменений, а не подтверждение публикации тега или контейнера.

### English

- Added Plane integration: protected settings, a meeting Wiki page, manual task and hypothesis delivery, and separate automation switches.
- Delivery state persists locally. An unknown POST outcome requires reconciliation; regenerated notes preserve remote content and manual Plane edits.
- Added Docker profiles for full speech processing and existing transcripts, separate persistent data and secret volumes, installation, and image updates.
- Recorded public speech dependencies and DiariZen compatibility changes. Added checks for builds, lightweight profile startup, and persistence across restarts.
- Updated the Russian and English READMEs, product diagram, installation guides, and documented limitations.

**When updating:** back up both volumes. Existing meeting notes are not automatically backfilled into Plane. Keep `plane.sqlite3` and the master key together for delivery recovery. Updates preserve paused processing; the explicit `--enable-speech` flag enables it on an existing installation. Use `--build` to include your own source changes. See the [Docker guide](docs/DOCKER.md) and [Plane guide](docs/PLANE.md).

The full speech environment passed build and library import checks. Model downloads, GPU audio processing, and summary quality need separate validation. This dated section records changes; it does not establish that a release tag or container has been published.

### Earlier experiments / Предыдущие эксперименты

The Gemini and Opus experiments are retained in closed, unmerged pull requests. The active summary workflow uses Luna with the source transcript. These links identify existing public code, not meeting transcripts, model outputs, or credentials.

Эксперименты Gemini и Opus сохранены в закрытых PR без объединения. Действующий маршрут использует Luna и исходную стенограмму. Ссылки ведут на уже публичный код; частные записи, ответы моделей и ключи сюда не добавлялись.

| Experiment / Эксперимент | Pull request | Preserved tip / Зафиксированный commit |
|---|---|---|
| Gemini audit and repair | [#3](https://github.com/R1venDev/TranscriSummaryzator/pull/3) | [`339c93b5244951862ae3a6a6fa1849efc8272e66`](https://github.com/R1venDev/TranscriSummaryzator/commit/339c93b5244951862ae3a6a6fa1849efc8272e66) |
| Opus Batch audit | [#4](https://github.com/R1venDev/TranscriSummaryzator/pull/4) | [`704fe439fd13f28e2c353d069ed06e3a3d027c5e`](https://github.com/R1venDev/TranscriSummaryzator/commit/704fe439fd13f28e2c353d069ed06e3a3d027c5e) |
