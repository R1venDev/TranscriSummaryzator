# Происхождение речевого Docker runtime

Снимок снят 2026-10-02 чтением metadata установленных Python-пакетов и Git diff
действующего Linux runtime. Серверные окружения, модельные настройки и сами веса
при этом не менялись. Файлы `docker/requirements-*-linux.txt` содержат точные
версии публичных пакетов, установленных в каждом окружении. Они дополняют
native requirements; старый `install.sh` не редактировался.

| Окружение | Python | Torch / Torchaudio | NumPy | Исходники |
| --- | --- | --- | --- | --- |
| GigaAM | 3.11.16 | 2.10.0+cu128 | 2.4.6 | `salute-developers/GigaAM` @ `7447938d791c4f3e643386ee22c33777004293a5` |
| DiariZen | 3.11.16 | 2.10.0+cu128 | 1.26.4 | `BUTSpeechFIT/DiariZen` @ `844f5555b0a98acd0931511fc641a8c5b8ba92c7` + сохранённый patch |
| Ultra/ReDimNet | 3.12.14 | 2.11.0+cu128 | 2.5.3 | `NVIDIA/NeMo` @ `abb8254dac2bf5a011e6069fcaa7df71c7e3b8c1` |

`uv==0.12.11` устанавливает две версии Python и три отдельных venv. Снимки
содержат полный набор установленных пакетов, поэтому установка идёт с
`--no-deps`, после чего `uv pip check` проверяет согласованность зависимостей.
На исходных трёх Linux-окружениях эта проверка прошла. Torch CUDA wheels берутся
с официального индекса `download.pytorch.org/whl/cu128`, остальные публичные
пакеты — из PyPI. Снимок фиксирует версии, но не hashes всех wheel-файлов;
сохранённый image digest фиксирует уже собранный результат.

## Сохранённые изменения DiariZen

`docker/diarizen-runtime.patch` — точный Git diff четырёх файлов относительно
указанного commit. SHA-256 patch:
`6b4b2e5494fe60fce445eed8c79688781f5e05c603b6b1e6f612bb1ce4ec45fc`.

Он сохраняет уже действующие изменения:

- вывод прогресса сегментации/эмбеддингов и передача точного количества говорящих;
- ограничение числа кластеров в VBx при заданном exact/max;
- загрузка legacy checkpoint с архитектурной metadata после изменения
  `weights_only` в новых Torch;
- совместимость с удалённым из нового Torchaudio типом `AudioMetaData`.

Docker build выполняет `git apply --check` перед применением. Новые исправления
алгоритма и изменения порогов в этот patch не добавлялись. Патч не переносит
веса, конфигурацию встреч, имена участников или серверные секреты.

## Build на отдельном диске

Установка и `docker-install.sh --speech --build` используют хранилище текущего Docker Engine.
Если на нём мало места, BuildKit можно запустить с собственным state на другом
диске. Такой способ не меняет `data-root` действующего Docker daemon. Замените
`/large-disk/transcri-build` на новый каталог на диске с достаточным местом:

```sh
mkdir -p /large-disk/transcri-build/buildkit
export TRANSCRI_SOURCE_REVISION="$(git rev-parse HEAD)"
export TRANSCRI_IMAGE=transcrisummaryzator:local-summary
export TRANSCRI_SPEECH_IMAGE=transcrisummaryzator:local-speech
docker run -d --name transcri-buildkit --privileged --cpus=4 --memory=12g --mount type=bind,src=/large-disk/transcri-build/buildkit,dst=/var/lib/buildkit moby/buildkit:v0.31.1
docker buildx create --name transcri-builder --driver remote docker-container://transcri-buildkit
docker buildx build --builder transcri-builder --platform linux/amd64 --target speech --build-arg SOURCE_REVISION="$TRANSCRI_SOURCE_REVISION" -t "$TRANSCRI_SPEECH_IMAGE" --output type=docker,dest=/large-disk/transcri-build/speech-image.tar .
```

BuildKit получает привилегии, необходимые для сборочных контейнеров. Его API
доступен через Docker socket без опубликованного TCP-порта. Нельзя подключать
к его state каталог с данными другого приложения. Этот способ описан в
[официальной документации remote driver](https://docs.docker.com/build/builders/drivers/remote/).

Docker archive можно перенести на подготовленный GPU-хост с местом для образа.
На нём используйте тот же checkout и значения трёх переменных выше:

```sh
docker load -i /large-disk/transcri-build/speech-image.tar
docker compose -f compose.yaml -f compose.speech.yaml build scheduler
docker compose -f compose.yaml -f compose.speech.yaml run --rm --no-deps --pull never bootstrap
docker compose -f compose.yaml -f compose.speech.yaml up -d --no-build --pull never --wait app scheduler
```

Если образ был собран с другим tag, задайте `TRANSCRI_SPEECH_IMAGE` с этим tag.
Обычный `docker-install.sh --speech` скачивает GHCR-образы; для использования
готового локального архива предназначены команды выше с `--no-build`.
Существующая настройка паузы обработки сохраняется. Для явного включения
речевой очереди добавьте `bootstrap --enable-speech` к команде запуска bootstrap.

После сохранения образа удалите только созданный для этой сборки builder:

```sh
docker buildx rm transcri-builder
docker rm -f transcri-buildkit
```

Архив и cache в `/large-disk/transcri-build` остаются до явного удаления
администратором. Не используйте глобальные команды prune для обновления приложения.

## Что подтверждает сборка

Последний этап Dockerfile выполняет `uv pip check` и импорт GigaAM, Silero,
Faster Whisper, DiariZen и NeMo Sortformer при отключённой сети. Модели не
создаются; аудио не обрабатывается. Это проверяет наличие библиотек и совместимость
импорта. Доступность GPU, загрузка заданных весов и качество всей речевой цепочки
требуют отдельной приёмки на целевом сервере.
