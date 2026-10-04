# Профили проектов / Project profiles

Одна установка поддерживает несколько самостоятельно создаваемых профилей: например, Aurion и ExLand. Откройте **Управление профилями** в верхней панели приложения, создайте профиль и выберите его. В его **Ключах моделей**, **Plane** и **Голосовых профилях** находятся отдельные настройки. Название можно менять без изменения идентификатора и существующих ссылок.

## Изоляция и совместимость

- Существующие записи, ключи, карточки, голосовые образцы и подключения остаются в профиле `default` (Aurion). Их файлы и идентификаторы сохраняются. Миграция добавляет поля, а не переносит старые credentials/generations.
- Новый профиль имеет случайный UUID, пустые ключи, своё подключение Plane (workspace/project/Wiki), свои переключатели автоматизации и голосовые образцы. Чужие ключи не используются как резерв.
- `jobs.project_id` закрепляется при загрузке. Браузер передаёт профиль явно в URL/заголовке; выбор в другом окне не меняет профиль фоновой задачи. Сервер проверяет принадлежность при просмотре, скачивании, редактировании и отправке. Изменить профиль уже отправленной записи через UI нельзя.
- `source_first_workflows.project_id` сохраняется до первого POST. Все следующие волны, чтение и cleanup Batch используют хранилище этого профиля, включая после перезапуска. Semantic reuse разделён по профилям. Схемы/модель/промпты не меняются.
- Очередь, scheduler и денежный ledger остаются общими: лимиты job/week не размножаются при создании профиля. OpenRouter account privacy/workspace verification продолжает действовать; иной OpenRouter workspace не разрешается просто добавлением ключа. Plane workspace задаётся независимо.
- Локальные task overrides и mapping, Plane outbox/media/settings и модельные credentials новых профилей находятся в `state/summary_private/projects/<id>/`. Секреты зашифрованы прежним master key; rotation и маскирование сохраняются.
- Голоса новых профилей находятся в `voice_profiles/projects/<id>/`, включая отдельный enrollment cache. ASR/диаризация, веса, thresholds, word IDs и исходные экспортные bytes не меняются. Новые встречи используют только голоса закреплённого профиля.
- Это профили одного администратора в закрытом Tailscale-приложении, не отдельные пользовательские аккаунты и не система multi-tenant permissions. Создание/переименование и операции с ключами защищены прежней административной проверкой.

## Native speech + Docker summary

При существующей native speech installation используется тонкий `scripts/native_projects_runner.py`: он импортирует установленный неизменённый pipeline, прежний адаптер удаления и добавляет только принадлежность jobs/голосов. `TRANSCRI_PROJECT_APP_DATA` указывает host data Docker. Native watcher читает **только явно загруженные новые app jobs** и ставит их в прежнюю speech queue с durable `native_job_id`. Точный source path/hash восстанавливает mapping после сбоя, без повторного ASR. Legacy native summary отключён в адаптере; суммаризация выполняется существующим Docker scheduler.

Docker scheduler при `TRANSCRI_NATIVE_SPEECH_ROOT` читает только mounted native queue/exports, проверяет project/content identity и копирует готовые transcript exports в app output перед summary. Mounts read-only. Явные операции смены спикеров/применения голосов блокируют summary до готовности новой source revision. Native receipt сохраняется до выполнения; неопределённый исход не повторяется автоматически и отображается как требующий восстановления. Старые native записи не импортируются автоматически. Native HTTP голосов остаётся в том же адресе `/profiles`; выбираемый профиль передаётся явно. Установка с speech Docker image использует профиль непосредственно в штатном pipeline и не требует native adapter.

Не копируйте пример host paths из чужой установки. Shared queue/registry требуют точечных ACL для двух существующих пользователей; summary_private/keys не выдаются speech пользователю. Рабочее окно для переключения native wrapper проверяется по отсутствию running/queued jobs. Активный ASR не прерывается.

## Release / rollback

До включения: сохранить compose и native service drop-in; SQLite backup queue/registry; metadata receipts без secrets. Сборка isolated release, offline namespace/ledger/HTTP tests и прежние speech-boundary tests. Не делать inference для теста изоляции.

Откат: вернуть прежний image/compose и native ExecStart, остановив только app/scheduler и переключив speech в безопасное окно. Новые `project_id` columns/registry/voice directories остаются, существующие источники и ручные правки не удаляются. Старый release **не изолирует новые профили**: до его запуска включите maintenance/закройте доступ к новым профилям и не обрабатывайте их очереди. Для обычного восстановления используйте исправленный release с поддержкой профилей, а не старые readers, игнорирующие ownership.

## English

Project folders isolate recordings, model credentials, local card edits, Plane workspace/settings/outboxes, and voice profiles within one installation. Create and select folders from the application toolbar. Existing data stays in Aurion; new folders start empty. Ownership is fixed at upload and persisted in Batch workflows. Shared accounting, privacy checks, the existing scheduler, source provenance and publication protocol remain in force. A folder switch cannot redirect an in-flight Batch or export. Profiles are not separate user accounts. Native speech routing changes ownership/storage only; recognition models and transcript content remain unchanged.
