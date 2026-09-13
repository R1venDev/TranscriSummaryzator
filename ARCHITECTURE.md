# Архитектура транскрипции и саммари

## v14: evidence ledger и MeetingState

Markdown-саммари теперь считается представлением, а не источником истины. Новые
запуски сохраняют исходные ASR-слова со стабильными `W########`, отдельное
нормализованное представление и точные `EvidenceSpan`. Исправления speaker ID
добавляются в историю resolution и больше не удаляют acoustic warning flags.

Семантический слой публикует `DialogueEvent`, `Relation` и `MeetingState` в
`semantics/`; решения, задачи и вопросы в `views/` являются его проекциями.
Extraction использует адаптивные turn-aware сегменты с контекстным halo, но
ссылаться на evidence разрешено только внутри текущей зоны.

`PipelineConfig` валидирует конфигурацию, запрещает неизвестные ключи и пишет
эффективные значения в `state/config.resolved.json`. Cache использует зависимости
конкретного stage, а очередь атомарно резервирует job с worker/attempt/lease.

## AS-IS (baseline preserved)

- `pipeline.py` owns the SQLite queue, resumable stages, web API and exports.
- FFmpeg creates one timestamp-preserving 16 kHz mono PCM working file.
- DiariZen Large s80 v2 produces anonymous speaker intervals and RTTM.
- GigaAM v3 produces Russian words with timestamps independently of diarization.
- Voice profiles are durable JSON records with reference WAV files and cached
  WeSpeaker embeddings. Post-processing attaches words to DiariZen intervals.
- Completed expensive stages are stored in `work/jobs/<job>/`; re-export does
  not repeat ASR or DiariZen.

The baseline is kept for benchmark comparison. Its main limitation is that one
anonymous clustering result is asked to provide both boundaries and identity.

## TO-BE

The queue, UI, preprocessing, DiariZen, GigaAM and exports remain in place.
New stages are additive and independently cached:

1. `audio.wav` — canonical 16 kHz mono audio, original timeline preserved.
2. `diarization.json` / `diarization.rttm` — DiariZen primary hypothesis.
3. `ultra.json` / `ultra.rttm` — Ultra Sortformer independent hypothesis.
4. `track_mapping.json` — temporal IoU/coverage matrix and optimal assignment.
5. `consensus.json` — atomic timeline with per-source evidence and overlap.
6. `redimnet_cluster_embeddings.json` plus the enrollment cache — ReDimNet2-B6
   robust global and session embeddings.
7. `debug.json` — automatic identity arbitration, session prototypes and
   selective second-pass decisions.
8. `result.json`, `result.rttm`, `debug.json` — identified final timeline.

Models execute sequentially on the RTX 5070 Ti. Each worker exits after its
stage, which releases its model and CUDA context. Cache metadata contains audio
hash, model revision and material configuration so post-processing changes do
not invalidate inference artifacts.

Unknown voices are first-class `UNKNOWN_n` identities. Diarizer slot/cluster
numbers never become known identities and are only retained in debug output.

## Delivery stages

- Stage 1: DiariZen + Ultra automatic inference.
- Stage 2: temporal track matching and consensus timeline.
- Stage 3: ReDimNet2 enrollment, robust prototypes and cluster Voice ID.
- Stage 4: automatic conflict arbitration and session prototypes.
- Stage 5: short-turn, overlap and selective second-pass corrections.
- Stage 6: domain benchmark and threshold tuning. DER requires human reference
  RTTM. `scripts/benchmark.py` compares any number of RTTM hypotheses and
  reports DER, confusion, missed speech, false alarm and overlap metrics.

All six implementation stages are present. Threshold tuning is intentionally
data-driven: no DER claim is made until reference RTTM is supplied. The public
entry point is `python diarize.py --audio meeting.wav --enrollment enrollment
--output results`; the web queue invokes the same automatic stages.

## Сквозная модель доверия

`transcript.json` версии 2 хранит неопределённость каждого слова и реплики по двум независимым каналам:

- распознавание: уверенность ASR, если модель её действительно возвращает, иначе `null` и источник `unavailable`;
- говорящий: эвристическая уверенность, причины риска и доля слов, для которых говорящий выведен из контекста.

Саммари копирует эти признаки в доказательства каждого тезиса и агрегирует их в `fact.uncertainty`. Маркер `⚠` в Markdown означает, что хотя бы одна подтверждающая реплика требует проверки. Эвристический балл говорящего нигде не называется вероятностью, а отсутствующая уверенность ASR не подменяется выдуманным числом.

## Семантический слой

После финальной проверки фактов отдельный проход создаёт `semantic_records.json`. Для каждого тезиса там независимо хранятся субъект, предикат, объект, модальность, условия, числа, автор высказывания, автор предложения, исполнитель, подтверждающие реплики и исходные `evidence_ids`. Все ссылки фильтруются по фактическим доказательствам тезиса.

Действия дополнительно экспортируются в `tasks.json`. Поле `tasks` содержит только записи, безопасные для автоматического создания; сомнительные действия остаются в `review_candidates` и не показываются как поручения. Назначение имеет статус `confirmed`, `unconfirmed` или `unknown`; автор высказывания не становится исполнителем автоматически. Подтверждение хранится и как идентификатор, и как отдельная запись с говорящим и текстом реплики. Для публичной задачи отдельно формируются `title` и `details`: модель получает подтверждённое действие и ближайший проверенный контекст, а независимый аудитор допускает формулировку только при наличии подтверждающих реплик. При сбое этого этапа остаётся исходная подтверждённая формулировка, поэтому публикация не прерывается.

У каждой опубликованной гипотезы показывается автор из `speaker_refs`. Технические фрагменты с неуверенным ASR или говорящим остаются в `summary_audit.json`; пользовательское саммари не выводит обезличенный перечень исключений.

Статус `confirmed` допустим только тогда, когда подтверждающая реплика принадлежит самому исполнителю. Согласие другого участника может подтвердить направление работы, но не принять задачу за назначенного человека. Для автоматического создания задачи дополнительно требуется отсутствие флагов неопределённости в подтверждающих репликах.

Перед публикацией единый расчёт метрик сверяет число тезисов, задач, подтверждений, нерешённых вопросов и элементов для проверки по аудио во всех итоговых JSON-файлах. Это исключает расхождение между видимым саммари, реестром задач и отчётом аудита после повторной проверки.

## Отказоустойчивость саммари

Каждый пакетный ответ проверяется на двух уровнях: JSON должен быть завершён, а набор идентификаторов ответа должен точно покрывать входной набор. При достижении лимита вывода или пропуске элементов пакет автоматически делится пополам. Одиночный тезис никогда не считается подтверждённым только из-за отсутствия ответа модели.

Проверка строится как сочетание независимых механизмов:

- генеративная модель извлекает и редактирует тезисы;
- отдельные промпты повторно проверяют критичные факты и итоговые формулировки;
- детерминированные правила контролируют evidence IDs, числа, отрицания, говорящих, исполнителей и подтверждения;
- неполный ответ модели не интерпретируется как отклонение факта;
- единичный неподтверждённый тезис исключается и сохраняется в аудите, а массовое отклонение блокирует публикацию;
- при сбое генерации раздела документа он собирается из уже проверенных атомарных фактов; при сбое синтеза заголовок строится из названий подтверждённых разделов.

Краткое описание формируется отдельным этапом после полной проверки фактов. Генератор получает компактную хронологическую опору из подтверждённых реплик и пишет ровно два связных абзаца: контекст и центральная проблема, затем подходы, выводы и следующие шаги. Независимый аудитор другого класса повторно сверяет каждый абзац с дословными репликами. Программа отдельно запрещает новые числа, неизвестные `fact_ids`, служебные оговорки и сомнительные фрагменты. Резервный вариант не склеивает разнесённые проблемы в новую причинную цепочку: первый абзац опирается на один надёжный тезис, второй — на явные следующие шаги. Если модель временно недоступна, точный детерминированный текст по проверенным тезисам позволяет завершить саммари без ручного редактирования.

Проверка устроена как каскад. Малая модель проверяет весь атомарный реестр, а отдельная модель 27B повторно проверяет только те тезисы, которые реально попадут в краткое описание, таймкоды, вклад участников, задачи, гипотезы, вопросы и подробную хронологию. Такой каскад не тратит часы на повторную проверку скрытых служебных записей. Если 27B временно недоступна, пайплайн завершает работу по уже проверенному реестру и записывает деградацию в аудит.

Таймкоды являются отдельным навигационным представлением. В них попадают только конкретные, самодостаточные и подтверждённые технические тезисы с достаточным лексическим подтверждением в исходной реплике. Ссылка привязывается к началу подтверждающей реплики, а не к времени, предложенному моделью. Организационные договорённости, вопросы без ответа, элементы для сверки с аудио, неопределённые говорящие, служебные пояснения и короткие фрагменты исключаются. Неуверенные фрагменты также не попадают во вклад участников и подробное описание: видимый отчёт сообщает только место и тип неопределённости, не повторяя сомнительное утверждение.

## Контроль изменений

`scripts/evaluate_pipeline.py` сравнивает baseline и candidate только с размеченными человеком материалами из `benchmark/gold`:

- WER/CER измеряют текст;
- DER, пропущенная речь, ложная речь и путаница говорящих измеряют diarization;
- precision/recall/F1 тезисов, типов, исполнителей и условий измеряют смысл.

Режим `--fail-on-regression` блокирует выпуск при ухудшении любого основного показателя. Автоматический вывод pipeline запрещено помечать как `gold`; до появления ручной разметки система сообщает отсутствие измерения и не публикует фиктивную «точность».
# Summary v16: canonical meeting intelligence

High-risk claims are verified by Ministral 3 14B. Critical claims require
matching verdicts from Ministral and Gemma 3 12B; disagreement fails closed
after the evidence-repair pass instead of being exposed in the final summary.

The publication path is now state-first:

```text
Audio → transcript + immutable word ledger → risk scheduler
      → atomic typed claims → evidence repair (critical windows)
      → independent validation → global dialogue relations
      → canonical MeetingState → deterministic views
      → constrained narrative → public-surface audit
```

`semantics/meeting_state.json` is the only semantic authority used by the
writer. `DecisionsView`, `TasksView`, `QuestionsView`, `TimelineView`, and
`SummaryView` are deterministic projections of that state. Markdown and HTML
only render those projections and cannot promote a proposal to a decision.

Every canonical event carries a stable `claim_id`, exact `evidence_ids`, source
word IDs, the source audio SHA-256, risk/compute policy, model provenance, and
typed condition/quantity/time components. Publication fails closed when a
claim cannot be traced to both immutable audio and ASR-word evidence.

The global relation resolver processes the whole meeting and emits stable
relations (`accepts`, `answers`, `contradicts`, `corrects`, `supersedes`, and
`accepted_by`). Narrative text may only introduce semantic connective language
when the cited claims have a corresponding relation; otherwise the generated
sentence is replaced by the canonical claim wording.

Risk controls compute instead of merely decorating output: LOW claims use
deterministic validation, MEDIUM claims add a verifier, HIGH claims use the
independent high-risk model with expanded context, and CRITICAL claims also
trigger a batched second ASR pass over `t−2s…t+4s` audio windows. Disagreement
on numbers or negation remains CRITICAL and is exposed to all later auditors.

`evaluate_pipeline.py` reports claim precision/recall, decision and action
precision, assignee and condition F1, deadline/number/negation/question
accuracy, unsupported-relation rate, citation precision, omission rate, and
end-to-end error attribution.

## Structured diagnostics

All processes append schema-versioned events to one per-job
`diagnostics.jsonl`. Decision events retain candidates, inputs, measured
values, thresholds, reasons, evidence references and the selected outcome.
Stage events retain wall-clock duration, cache selection, model metadata and
failures. Writes use one `O_APPEND` system call, so isolated ASR/diarization
environments can safely contribute to the same ledger.

`diagnostics_summary.json` contains counts by component/category/severity and
the latest error plus the JSONL digest. Both artifacts are copied to the public
job output after transcription, after summary completion and after summary
failure. They are downloadable from the UI but remain excluded from Git.
