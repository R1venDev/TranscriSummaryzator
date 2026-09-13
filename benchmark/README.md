# Эталонная проверка pipeline

## Release gates v20

Gold scorecard включает atomic claim precision/recall, canonical type accuracy, relation precision, modality/latest-state/correction accuracy, task acceptance F1, question slot/direct-answer accuracy, mandatory recall, episode/topic coverage, cross-episode merge error, redundancy, speaker ECE/Brier и critical ASR number/negation errors.

Release блокируется при регрессии critical factual precision или mandatory recall, любом известном cross-episode merge, превышении порога false-positive задач/false-resolution вопросов либо недопустимой деградации DER/JER. `evaluation/hard_negatives.py` генерирует corruption pairs для speaker, number, negation и modality; набор расширяется assignment, answer-swap, causal inversion и supersession примерами по мере пополнения gold meetings.

Эта папка предназначена только для данных, вручную сверенных с аудио. Автоматическую транскрипцию нельзя копировать в reference и помечать как `gold`.

Для каждого случая нужны:

- дословная эталонная транскрипция в JSON или TXT;
- RTTM с ручными границами и говорящими;
- `semantic_records.json` с ручным списком утверждений, задач, исполнителей и условий;
- для каждой проверяемой системы — явное сопоставление её тезисов с эталонными в `alignment.json`.

Manifest поддерживает несколько записей и несколько систем. Поля `baseline` и `candidate` включают расчёт изменений. Запуск:

```sh
python3 scripts/evaluate_pipeline.py --manifest benchmark/gold/manifest.json --output benchmark/results/latest.json
```

Для проверки перед выпуском добавьте `--fail-on-regression`: команда завершится ошибкой, если WER/CER/DER выросли либо F1 смысла, исполнителей или условий снизился относительно `baseline`.

Метрики считаются независимо: WER/CER для текста, DER и его компоненты для говорящих, precision/recall/F1 для смысла, исполнителей и условий. Значения macro усредняются по встречам; отчёт также хранит результаты каждого случая.

Рекомендуемый первый набор: не менее трёх встреч и 30–60 минут суммарного аудио, включая перебивания, короткие ответы «да/нет», числа, сроки и распределение задач. Часть встреч используется для настройки, другая остаётся закрытым test-набором.
