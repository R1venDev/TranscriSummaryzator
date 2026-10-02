# Luna source-first через OpenRouter Batch: устройство и выпуск

## Адаптивный output и пригодные частичные отчёты — 3 октября 2026

Policy `luna_batch_source_first_v2`, prompts `luna_batch_stage_v2`. Штатный маршрут остаётся Luna/OpenRouter Batch на единственном разрешённом OpenAI endpoint. ASR, диаризация, входная стенограмма и речевые окружения не изменены.

В новом маршруте нет таблицы фиксированных 25k/32k caps. `capacity.py` вычисляет верхний `max_tokens` для каждого реального payload из подтверждённого endpoint output limit, оставшегося контекста, тарифов без обещанного cache hit и оставшихся разрешённых денег. Из денег сначала вычитаются фактические input holds текущей волны и плановая стоимость известных будущих source inputs; output-деньги делятся между текущими и оставшимися items. Это резерв для reasoning и видимого JSON вместе, не требуемый размер ответа. После получения фактических bills освобождённый запас доступен следующим стадиям. Максимальное окно модели не резервируется целиком.

Первоначальный input forecast — нижняя плановая стоимость известного источника, а не обещание цены всей цепочки: будущие отчёты ещё неизвестны. Каждая следующая волна резервирует реальные bytes и вычисленную output capacity до POST. При нехватке денег — честный `budget_blocked`, без повышения лимита, усечения source или sync. Действующие разрешения: $0.25/job, $5 за скользящие 7 суток, до 12 items, 6 Batch POST. Подтверждённые ограничения модели/API и деньги остаются конечными. Ограниченный split/retry по `length` пока не реализован; повреждённый JSON не принимается. Старые legacy jobs читаются прежними readers; legacy inference не является fallback.

Allocation сохраняется до intent и POST с identity исходных данных; перезапуск использует те же body bytes и custom IDs. Перед отправкой сохранённые caps повторно проверяются по текущему endpoint и тарифу. Изменение prompts/policy входит в semantic identity: старый непроверенный output не становится cache hit новой policy.

### Исправления и проверка полноты

- Audit может ссылаться на проверенную `SOURCE_EVIDENCE` из своего входа. Эти цитаты повторно проверяются по той же source revision. Новая дословная цитата того же U-ID может получить report-local scope с provenance; перенос существующего ID на другую реплику запрещён.
- Независимые валидные findings/needs_context сохраняются даже при неполной typed-проверке. Неверные цитаты, неизвестные/двойные IDs удаляются с причиной в приватном normalized report. Salvage не подтверждает положительное покрытие.
- Global использует только app-issued link IDs. Ошибочные или отсутствующие link/resolution verdicts не превращаются в `supported`, но не уничтожают независимые grounded findings.
- Repair получает D0, полный source, findings, таблицу source evidence и app-issued surface targets. Не более одной смысловой итерации; patch применяется к временной D1.
- Verify принимает только явно одобренные, локально валидные bundles. Отсутствующее verdict, неверная цитата, reject или unresolved откатывают зависимую группу. Неполный общий report не отменяет явно проверенную независимую группу. Новое замечание откатывает затронутые группы; без координат затронутого места все группы остаются неприменёнными.
- Публикация собирает один canonical document из принятых patches и явных пометок неопределённости, затем renderer/task projection и atomic pointer. D0/raw сохраняются. Task UUID, CAS и ручные overrides остаются в существующем TaskStore. UI показывает число применённых bundles отдельно от предупреждения о неполноте.

### Выполненная проверка и её предел

На сохранённых ответах реальной встречи из 411 реплик offline replay восстановил 7 findings аудита и 5 findings global. Два обрезанных `length` ответа не восстановлены. D0 не изменён. Эти 12 замечаний — кандидаты Luna, а не доказанная истина и не уже применённый live repair. Unit/integration проверки включают получение evidence по входному ID, плохой соседний finding, неправильный link ID, неполную verification и цепочку repair→verify→apply. Качество новой генерации требует отдельного полного API-прогона; offline PASS не измеряет смысловую точность.

Официальные источники, сверены 02–03.10.2026: [OpenRouter reasoning/output](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [Batch contract](https://openrouter.ai/docs/batch-quickstart), [OpenAI reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning-best-practices). Context7 `/openrouterteam/docs` использован для адресной сверки; фактический endpoint Chat serializer имеет приоритет над общими примерами.

### Включение и откат

Новая policy включается существующим `TRANSCRI_LUNA_SOURCE_FIRST_ENABLED=1`, с сохранённым `TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD=0.25`. Отдельный scheduler/ключевой store не создаётся. До смены summary image сохраняются точные compose/digest и checkpoint активных jobs. При отсутствии активных jobs меняются только контейнеры app/scheduler; speech service не перезапускается. Откат — прежний pinned summary digest с теми же томами. Уже отправленные Batch сначала доводятся штатным scheduler новой версии до terminal: старый release не должен отправлять повтор или продолжать job с изменённым prompt. Прежние sealed generations и ручные правки сохраняются.

## Восстановление недоступного Batch — 3 октября 2026

После `202 Accepted` GET может перестать возвращать зарегистрированный Batch. HTTP 404 не доказывает отсутствие генераций или списания. Worker сохраняет remote ID, исходные item-слоты и весь unknown hold; повторный POST этого пакета запрещён.

Два наблюдения 404 с интервалом не менее пяти минут, свежим подтверждением прежнего workspace и полным списком метаданных в окне отправки переводят попытку в **локальное** `remote_unavailable`. Это не upstream `failed`/`cancelled`: remote status и bill остаются неизвестными. Неполный список, другой workspace или transient 404 не разрешают такой переход; успешный GET сбрасывает подтверждение отсутствия. Свои receipts сохраняются приватно, чужие workspace jobs не скачиваются и не изменяются.

Недоступный audit не блокирует существующие global/repair/verify стадии. Они используют сохранённый целый D0, пригодный inventory и полный source, внутри прежних лимитов. Публикуются только явно принятые исправления; отсутствие audit остаётся `review_incomplete`, независимо от успеха global. Если поздней стадии не хватает бюджета, действует существующая публикация с оговорками. Writer без пригодного целого по-прежнему не превращается в выдуманный конспект.

Фактический ответ нового Batch create сохраняется в `submissions/<attempt_id>.json` после durable записи remote ID. Capture failure не разрешает повтор POST. Gateway cleanup теперь выполняется после окончания logical workflow, только для terminal попыток с проверенным локальным raw. Отдельный unknown hold не освобождается из-за публикации, рестарта или возраста. Для дальнейшего billing reconciliation сохраняются ID и диагностические SHA; автоматического освобождения денег по 404 нет.

Миграция ledger аддитивна: nullable missing timestamp/evidence и счётчик наблюдений. Prompts, WriterSchema, source identity, $0.25/job, $5/rolling week и максимум 12 items не меняются. При откате старый worker может не понимать `remote_unavailable`: он не должен пересылать этот пакет; сохранённую generation можно читать, а завершение новых стадий следует возобновить на совместимом release. Источники, ручные edits и история остаются на томах.

Контракт повторно сверён через Context7 `/openrouterteam/docs` и первичный [OpenRouter Batch Quickstart](https://openrouter.ai/docs/batch-quickstart): 202 означает принятие и сохранение, GET выдаёт inline results, listing scoped к workspace, DELETE относится к конкретному terminal Batch. Документация не объясняет исчезновение текущего remote ID; причина не объявляется установленной.

## Историческое состояние v1

Следующие сведения относятся к прежним native releases и опытам; они не доказывают текущий Docker rollout.

## Исходное состояние перед правками

Первый инженерный worktree `codex/luna-batch-production-20260928` создан из чистого checkout `b5881294ff7ddbf4654adafac63e6d94197d97fb`. У него действовали WriterSchema `output_schema_v1.json` (SHA-256 `652ffa8f8ad65e1994092bae9222fe172e13c9b8b895d4735abf85c64968edc7`), Luna prompt `prompt_v1.md` (SHA-256 `4e9fbf912ea3f7fc8d7ba97495a2eb59d5715b07a2cea045466ceef3b49d2c01`) и legacy default quality policy `claude_opus_5_5_partitioned_audit_v3`. До изменений этот checkout не имел незакоммиченного diff. Feature diff перенесён на актуальный GitHub `main` `4276dfcfb3f2087886895a30f5f376a68531995e`; удалённые на `main` исследовательские Gemini/Opus модули не возвращались.

На Linux `transcri-luna-dashboard` и `transcri-luna-summary-scheduler` запускают `pipeline.py` из `/mnt/shared-data/transcri-work/summary-luna-source-first-r1-20260928`, tree `b70d4c5`. Каталоги `inbox`, `outputs`, `state`, `work` остаются общими с предыдущим summary release. Старые drop-in `50-gemini-candidate.conf` сохранены как путь отката; текущие `60-source-first.conf` задают `TRANSCRI_LUNA_SOURCE_FIRST_ENABLED=1`, предел $0.25/job и проверенный workspace ID без ключа.

## Выполнение

Штатный `pipeline.py` вызывает `scripts/luna_summary_worker.py`; его `submit` и `poll` используют `summary/luna_v1/source_first_runtime.py`, существующий `CredentialStore`, `Ledger`, source resolver, `TaskStore` и `publish_document`. `summary_backend` в [примерной конфигурации](../config.example.json) пока равен `legacy_local`. Новый job закрепляет SHA исходного `transcript.json`, версии policy/prompts/schemas, workspace и route в неизменяемом manifest. Стадии независимы по контексту; ответ одной стадии попадает в следующую только после локального сохранения и проверки.

| Волна | Работа | Batch create POST | Items при K=M=3 |
|---|---|---:|---:|
| 1 | S1W: один full-source writer; параллельно S1E: K source-only извлечений | 2 | 1 + 3 |
| 2 | S2: локальная нормализация и реестр surfaces; S3: M двунаправленных проверок | 1 | 3 |
| 3 | S4: один full-source global check связей, поправок и `needs_context` | 1 | 1 |
| 4, при findings | S5: один адресный repair | 1 | 1 |
| 5, при bundles | S6: одна проверка D0, D1 и diff; затем S7: локальная публикация | 1 | 1 |

Итого для K=M=3: **4–6 Batch create POST и 8–10 inference items**. Это разные счётчики; один POST может содержать несколько items. Верхний предел — 12 потенциально оплачиваемых items; при K=M=3 остаются два дополнительных слота, при K=M=4 все 12 заняты штатными стадиями. Прогноз и hold учитывают только исполняемые стадии; два пока не реализованных recovery-слота не резервируются как будущий расход. Планировщик строит K/M из фактического объёма source, не обрезая стенограмму. Если нужное число пакетов превышает 12 items, job получает `dimension_budget_blocked`. Восстановление output-limit split/retry пока не реализовано: проблемный item фиксируется как неполный, writer без целого результата не публикуется, а пригодный D0 при сбое дальнейшей стадии показывается с `review_incomplete`.

Writer сохраняет прежний [WriterSchema](../summary/luna_v1/contract.py); названия API-полей не совпадают с восемью заголовками [renderer](../summary/luna_v1/render.py). Источник имеет уникальные U-ID даже при одинаковых таймкодах. Навигационный диапазон главы и связанная с ней поздняя цитата могут различаться. Карточка с содержательными `title` и `description` остаётся видимой при неизвестных `assignee`, `due`, `priority` или `recipient`; статус обсуждения отделён от статуса проверки модели. UUID/якоря задач и пользовательские edits принадлежат приложению, а не модельному выводу.

## Запрос и контракты

Единственный модельный транспорт новой policy — `POST https://openrouter.ai/api/v1/batches`. Сохранённый перед резервом конверт отправляется теми же байтами: top-level поля `endpoint`, `model`, `provider`, `completion_window`, затем `requests`. Их значения: `/v1/chat/completions`, `openai/gpt-6-luna`, `{ "only": ["openai"] }`, `24h`; каждый item имеет уникальный `custom_id`, `messages`, свою schema, reasoning и output cap. Суффикс `:batch` обозначает проверяемый Batch endpoint/тариф в каталоге, а не вторую скидку. `endpoint` внутри конверта задаёт формат Chat item и не разрешает sync вызов. При отказе Batch нет sync, другой модели или другого провайдера.

`verify_source_first_batch_route` проверяет inference key, workspace, разрешённую модель, единственный OpenAI Batch endpoint, параметры, capacity и текущий тариф до приватного POST. Ключ, которому Luna не разрешена, исключается при выборе Luna-credential; другие route/privacy/budget ошибки не запускают перебор ключей. Привязка проверенного workspace задаётся `TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID`; сам `GET /api/v1/key` не подтверждает account privacy/region policy. Если обязательную характеристику endpoint или privacy-контроль нельзя доказать, отправка блокируется. Текущий каталог OpenAI Batch endpoint объявляет Chat-параметр `max_tokens`, поэтому новая policy передаёт именно его (верхний cap включает reasoning); общий gateway API также документирует `max_completion_tokens`, но два алиаса одновременно не посылаются. В Batch `provider` нельзя переносить sync-настройки `allow_fallbacks`, `order`, `sort`, `require_parameters`, `zdr`.

Версия промптов — `luna_batch_stage_v1`: [общий текст и файлы writer/extract/audit/global/repair/verify](../summary/luna_v1/prompts_batch_v1/00_common.md). [Типизированные sidecar-контракты](../summary/luna_v1/batch_stage_contracts_v1.py) и выгруженные JSON Schema inventory/audit/global/patch/verify/review — в [schemas_batch_v1](../summary/luna_v1/schemas_batch_v1/inventory_v1.json). Runtime передаёт общий developer-текст плюс одну стадию, сериализованный JSON данных в user message и только схему этой стадии. Для writer используется настоящий WriterSchema. Поля wire-схем обязательны, неизвестные business values nullable, лишние поля запрещены. Строковые U-ID, membership, цитаты, source hash, бюджеты, task identity и право на публикацию проверяет код. Отчёт Luna остаётся оценкой, а не ground truth.

| Стадия | Reasoning | Верхний `max_tokens` для подтверждённого Batch Chat endpoint |
|---|---|---:|
| Writer | medium | 32 000 |
| Extraction | medium | 25 000 на item |
| Audit, global, repair, verify | high | 25 000 на item |

Лимит включает billed reasoning и видимый ответ; это cap, а не размер, к которому надо стремиться. Запросы не содержат tools, web, MCP, stream, temperature, top_p или logprobs. Модель должна возвращать проверяемые evidence и краткий вывод, без скрытой цепочки рассуждений. Для audit каждое ожидаемое source/surface ID получает результат; отсутствующее не трактуется как `supported`. Локальная проверка inventory подтверждает координаты и цитаты, но не объявляет интерпретацию верной. `needs_context` переходит в полный S4; поздняя поправка меняет лишь доказанно связанное утверждение.

## Durable recovery и публикация

`Ledger` хранит workflow, immutable Batch intent и envelope hash, expected `custom_id`, attempt, credential/workspace, remote batch ID, lease, следующий poll, raw/usage и денежные holds. Существующий немодельный scheduler делает GET только по наступлении `next_poll_at`; перезапуск продолжает тот же remote ID. После неизвестного исхода create POST используется workspace list и GET кандидата с проверкой его фактических item IDs. Неоднозначное совпадение остаётся `submission_unknown` с удержанным резервом; локальный lock не гарантирует exactly-once у gateway.

Завершённый Batch разбирается по `custom_id`, а не порядку `results`. Item error, refusal, `length`, missing, duplicate и лишние IDs остаются отдельными исходами; `completed` у Batch не превращает их в успех. Текущий OpenRouter GET `/api/v1/batches/{id}` возвращает завершённые результаты inline. Для failed/expired/cancelled `results` может быть `null`; доступные raw/usage сохраняются, отсутствующие outcome и bill остаются неизвестными. Никакого выдуманного `/results` или `output_file_id` у этого gateway нет.

В живом Batch GET для отправленного alias `openai/gpt-6-luna` наблюдался terminal `model=openai/gpt-6-luna-20260922`. Такой ответ принимается только при точном совпадении с `canonical_slug` актуальной карточки `openai/gpt-6-luna:batch` и однозначном OpenAI endpoint; совпадение префикса семейства недостаточно. Remote batch ID, Chat endpoint и `custom_id` каждого item по-прежнему сверяются с запечатанным намерением. Проверенная привязка сохраняется локально вместе с terminal evidence.

Repair состоит из минимальных typed bundles с source evidence, app-issued target и expected-before hash. Только принятые S6 bundles применяются к D0; reject, unresolved и отсутствующий verdict откатывают связанную группу. Известное спорное утверждение получает локальную оговорку. При неполном review целостный D0 показывается с `review_incomplete`; при отсутствии пригодного writer output UI показывает источник и `generation_failed`. Нового смыслового цикла нет.

Публикация связывает canonical JSON, `review_sidecar.json`, Markdown, HTML и task projection с одним sealed generation. `TaskStore` сохраняет ручные изменения как отдельную ревизию; конфликт CAS не переписывает их. Перед переключением current pointer файлы sealed и проверены. Принятое поколение из изолированной приёмки можно локально применить к штатной папке той же стенограммы без нового Batch: проверяются source SHA, workspace/policy, manifest, хеши всех файлов, удостоверение writer Batch и доступность тех же карточек в общем `TaskStore`. Копия публикуется под lock, current pointer меняется последним; старая generation остаётся для отката. Активная или несовпадающая попытка не переносится. Перед релизом требуется failure injection на границе task transaction и файлового pointer: отсутствие mixed generation нельзя утверждать только по локальному lock.

## Деньги и хранение

Сохранённый default этой policy — **$0.10 на logical job** (`JOB_CAP_MICROUSD=100_000`). 28 сентября 2026 пользователь разрешил **$0.25/job для новой policy**; для применения на Linux нужно явно установить `TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD=0.25`. Код ограничивает эту переменную верхним порогом $0.25 и отвергает более высокое значение. В тот же день пользователь поднял **общий лимит за скользящие семь дней до $5.00 across keys** (`WEEK_CAP_MICROUSD=5_000_000`); ранее он составлял $1.00. Недельный лимит общий для legacy и новой policy, а повышение не меняет их отдельные per-job caps. Значение каждого job запечатывается в его manifest/hold и не меняется при последующей смене окружения. Если прогноз полного плана или атомарный hold конкретного Batch превышает сохранённый предел, job блокируется до POST; его нельзя разбить на другие ключи ради обхода лимита. Показывать отдельно прогноз, hold, фактический bill и unknown. Batch total и per-item costs сверяются и учитываются один раз; cache hit заранее не предполагается. Тариф, long-context, регион, cache write и BYOK проверяются по фактическому endpoint перед допуском. [Карточка Luna Batch](https://openrouter.ai/openai/gpt-6-luna:batch/) даёт ориентир, не receipt этого checkout.

Batch хранит вход и результат у OpenRouter до 30 дней; `store=false` не означает ZDR. Отдельный gateway I/O logging оставлен включённым пользователем и может хранить полные входы/ответы не менее трёх месяцев. После terminal выполняются durable local capture и разрешённый DELETE: это подтверждено для четырёх Batch полного приёмочного прогона. DELETE не отменяет активную генерацию и не удаляет автоматически I/O или billing logs.

## Включение и откат

Перед включением сохранены прежние drop-in и generation pointer. На Linux после исправления прошли 577 unit/integration tests, включая Batch-only transport, recovery, task edits и publication. Mocks проверяют механизм, не смысловую точность модели. Live receipt: четыре Batch, 8/8 terminal items, $0.043585, `review_incomplete`; текущий видимый конспект сохранён из предыдущего поколения.

Новые jobs получают `luna_batch_source_first_v1`; уже созданные generations и закреплённые старые jobs сохраняют свою lineage. Dashboard и scheduler используют один новый release, speech не переключался. Первый прогон показал дефект валидации отдельных цитат и содержательные пропуски в D0; исправление проверено тестами, но новый платный прогон после него ещё не выполнен. До него старый production pointer остаётся на `20260926-184644-4f91eeefa5f3`.

При дефекте удалить только `60-source-first.conf` у двух summary units, выполнить `systemctl daemon-reload` и штатно перезапустить только `transcri-luna-dashboard` и `transcri-luna-summary-scheduler`; сохранённые `50-gemini-candidate.conf` вернут прежние release. Speech не перезапускать. Уже отправленный Batch продолжить безопасно сверять по его ID; неизвестный POST не повторять и не отправлять ту же встречу sync. Текущий generation pointer не менялся; при будущем откате выбирать предыдущую sealed generation вместе с её task revision, сохраняя пользовательские edits.

Официальный gateway contract: [OpenRouter Batch Quickstart](https://openrouter.ai/docs/batch-quickstart), [Chat completion](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion), [endpoint metadata](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model). Предел Chat output и reasoning: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create); модель: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna); приватность: [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data), [OpenRouter I/O logging](https://openrouter.ai/docs/guides/features/input-output-logging).
