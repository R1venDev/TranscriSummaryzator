# Установка в Docker

Docker-упаковка создаёт отдельную установку. Код остаётся в вашем Git checkout;
данные, записи, кэш моделей и секреты сохраняются в именованных Docker volumes.
Работающие systemd-службы и их каталоги эта установка не подключает.

## Что входит в профили

| Профиль | Содержимое | Условия |
| --- | --- | --- |
| `--speech` (по умолчанию) | Веб-интерфейс, GigaAM, DiariZen, Ultra/ReDimNet, FFmpeg, Luna/OpenRouter Batch, очередь и настройки интеграций | Linux amd64, NVIDIA GPU, совместимый драйвер и NVIDIA Container Toolkit |
| `--summary` | Веб-интерфейс, импорт готовой стенограммы, Luna/OpenRouter Batch, очередь и настройки интеграций | Linux-контейнеры; готовый `transcript.json`; API-ключ и проверенный workspace для отправки |

Речевой образ воспроизводит проверенный список установленных версий действующего
Linux runtime: Python 3.11/Torch 2.10 CUDA 12.8 для GigaAM и DiariZen,
Python 3.12/Torch 2.11 CUDA 12.8 для Ultra/ReDimNet. Три окружения изолированы.
Существующие изменения DiariZen сохранены отдельным patch с базовым commit.
[Точные версии и происхождение](DOCKER_RUNTIME.md) записаны вместе с исходниками.
Текущий статус сборки и проверки находится в конце этой страницы.

Веса моделей не входят в build context и не загружаются при сборке. В речевом
профиле их загружает существующий конвейер при первой обработке записи. Для
закрытых моделей потребуется соответствующий доступ Hugging Face. Объём GPU
пакетов и моделей значителен; выделите отдельный диск с запасом в десятки ГБ.
Для полной сборки планируйте не менее 60 ГБ свободного места под Python/CUDA
пакеты, build cache и образ, дополнительно — место под модели и записи. Точные
требования к VRAM подтвердите на своей длительности записи и GPU.

## Две команды для новой установки

Предварительно установите Git и Docker Engine с Compose v2.20+ либо актуальный
Docker Desktop. Для речевого профиля настройте NVIDIA Container Toolkit по
[официальной инструкции Docker](https://docs.docker.com/compose/how-tos/gpu-support/).

```sh
git clone https://github.com/R1venDev/TranscriSummaryzator.git && cd TranscriSummaryzator
./scripts/docker-install.sh --speech
```

Откройте **http://127.0.0.1:8765**. Установщик покажет логин `admin` и случайный
пароль **один раз в терминале**. Сохраните пароль в менеджере паролей. Повторный
запуск установщика сохраняет ключ шифрования, пароль и пользовательский config.
Установщик скачивает готовые образы `ghcr.io/r1vendev/transcrisummaryzator:speech`
и `:summary`. Если нужный образ недоступен, установка завершается ошибкой.
Для явной сборки своего checkout добавьте `--build`.

Речевой образ содержит тяжёлые окружения. В новой установке `--speech` включает
обработку аудио. При повторной установке или обновлении сохранённое
`processing_enabled` остаётся прежним, включая поставленную на паузу очередь.
Для явного включения обработки в существующем summary volume используйте
`./scripts/docker-install.sh --speech --enable-speech`. Параметры моделей,
устройств и пороги остаются из config.
Этот профиль нужно принять на целевом GPU перед использованием для рабочих
встреч. Для запуска только веб-интерфейса и конспектов из готовых стенограмм
используйте `./scripts/docker-install.sh --summary`; он не обрабатывает аудио.

По умолчанию опубликованный порт доступен только с Docker-хоста. Прямой HTTP
предназначен для локального браузера. Для удалённого доступа используйте SSH
туннель либо HTTPS reverse proxy и явно задайте его origin. Публичные страницы
стенограмм сохраняют существующую политику доступа приложения: не открывайте
порт напрямую в интернет без внешней защиты.

## Первый конспект из готовой стенограммы

1. Откройте `/summary-settings` и добавьте inference-ключ OpenRouter. Проверка
   ключа сама по себе не подтверждает право на Batch-модель или настройки аккаунта.
2. Проверьте политики своего OpenRouter workspace и задайте его точный ID:

   ```sh
   export TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID='your-verified-workspace-id'
   docker compose up -d
   ```

   Для речевого профиля используйте в командах Compose оба файла:
   `docker compose -f compose.yaml -f compose.speech.yaml ...`.
   Чтобы сохранить настройку между терминалами, передавайте переменную из своего
   менеджера конфигурации либо файла окружения за пределами Git. Это ID, не API-ключ.
   Без совпадения проверенного ID с metadata ключа отправка блокируется.
3. Импортируйте **существующий** файл формата TranscriSummaryzator:

   ```sh
   docker compose exec -T app python /app/docker/import-transcript.py --name 'Планирование 02.10.2026' < /path/to/transcript.json
   ```

   Импорт проверяет структуру и таймкоды, сохраняет исходные байты, помечает запись
   как импортированную и не запускает ASR или платный запрос. Повтор тех же байтов
   под тем же именем возвращает существующую запись. Аудиофайл не импортируется.
4. Откройте запись в dashboard и нажмите создание саммари. Состояние отправки,
   результаты проверок, бюджет и редактируемые карточки сохраняются после
   перезапуска контейнеров.

Compose включает существующий маршрут `luna_batch_source_first_v1` с лимитом
$0.25 на workflow и общим лимитом $5 за семь дней в durable ledger. Ошибка
не переключает обработку на локальную модель. Условия хранения Batch у
провайдера и проверку workspace см. в
[SUMMARY_ADMIN_KEYS.md](SUMMARY_ADMIN_KEYS.md) и
[SUMMARY_LUNA_SOURCE_FIRST.md](SUMMARY_LUNA_SOURCE_FIRST.md).

## Устройство установки

| Место | Содержимое |
| --- | --- |
| `/app` в образе | Код, UI, шаблоны; файловая система только для чтения |
| volume `transcri_data` → `/data` | `config.json`, `inbox/`, `outputs/`, `voice_profiles/`, очередь SQLite, приватные ledger, временные задания и `work/cache/` с моделями |
| volume `transcri_secrets` → `/secrets` | Master key Fernet и scrypt verifier администратора; доступ `0600`, владелец UID 10001 |
| `app` | Dashboard в summary-профиле; dashboard и речевой watcher в speech-профиле |
| `scheduler` | Единственный отдельный процесс `summary-scheduler` из лёгкого образа |

Префикс томов зависит от Compose project name; default — `transcri`. Для второй
независимой установки задайте другой `COMPOSE_PROJECT_NAME` и порт.

Контейнеры работают от UID/GID `10001:10001`, без Linux capabilities и с
`no-new-privileges`. `/tmp` — отдельный ограниченный tmpfs. Bootstrap получает
том секретов на запись; обычные процессы монтируют его только для чтения.
Открытый пароль не сохраняется. Inference-ключи и токены интеграций шифруются
приложением; сами тома Docker не являются шифрованным хранилищем. Администратор
Docker-хоста имеет доступ к обоим томам.

Scheduler использует существующий файловый lock, SQLite ledger и ID удалённых
Batch-заданий. После рестарта он продолжает известный запрос; неопределённая
отправка остаётся на восстановление и не становится новым POST автоматически.
Запускайте один `app` и один `scheduler`; это установка на одном Docker-хосте,
а не распределённая очередь. Не подключайте один том к двум Compose-проектам.
См. [сохранение Docker volumes](https://docs.docker.com/engine/storage/volumes/).

## Порт и HTTPS

Например, для другого локального порта:

```sh
export TRANSCRI_PORT=8767 TRANSCRI_PUBLIC_ORIGIN=http://127.0.0.1:8767
./scripts/docker-install.sh --summary
```

Меняйте одновременно `TRANSCRI_PORT` и `TRANSCRI_PUBLIC_ORIGIN`: административные
запросы проверяют точный Host/Origin. Для HTTPS proxy укажите
`TRANSCRI_PUBLIC_ORIGIN=https://your-host.example`. Список
`TRANSCRI_ADMIN_ALLOWED_PEERS` задаёт допустимые IP/CIDR непосредственного peer
в Docker-сети. Default покрывает обычные Docker bridge-сети
`172.16.0.0/12,192.168.0.0/16`; при собственной сети задайте её более узкий CIDR.
Заголовки с адресом клиента не заменяют Basic-аутентификацию.

## Изменение кода и обновление

Чтобы скачать новые опубликованные образы и пересоздать контейнеры с прежними томами:

```sh
./scripts/docker-update.sh --speech
```

Редактируйте Python/HTML/JS прямо в checkout. Для сборки этих изменений добавьте
`--build`; локальные образы получают теги `transcrisummaryzator:local-speech`
и `transcrisummaryzator:local-summary`:

```sh
git pull --ff-only && ./scripts/docker-update.sh --speech --build
```

Для лёгкой установки замените `--speech` на `--summary`. Установщик не выполняет
`git reset`, не удаляет локальные изменения и не заменяет сохранённый config.
Build сохраняет revision исходников в image metadata; пользовательские
изменения должны быть закоммичены для точного воспроизведения release.
Контейнеры пересоздаются с теми же volumes.

После локальной сборки для прямых команд `docker compose up` задавайте те же
локальные теги через `TRANSCRI_IMAGE` и `TRANSCRI_SPEECH_IMAGE`. Иначе Compose
использует опубликованные каналы по умолчанию. Команды install/update с
`--build` задают локальные теги сами.

### Фиксированная версия

Каналы `:speech` и `:summary` обновляются после проверки нового `main`.
Каждый опубликованный commit имеет отдельные теги
`:sha-<полный Git SHA>-speech` и `:sha-<полный Git SHA>-summary`. Workflow не
пересобирает существующие теги SHA и не перезаписывает release-тег другим образом.
Версионные теги вида `:2026.10.02-speech` создаются при публикации release.
Установщик сверяет revision обоих скачанных образов до bootstrap. Если каналы
попали в короткий промежуток между двумя обновлениями тегов, установка
останавливается; повторите команду после окончания публикации.

Для фиксации конкретного проверенного release задайте обе ссылки до установки:

```sh
export TRANSCRI_IMAGE=ghcr.io/r1vendev/transcrisummaryzator:2026.10.02-summary
export TRANSCRI_SPEECH_IMAGE=ghcr.io/r1vendev/transcrisummaryzator:2026.10.02-speech
./scripts/docker-install.sh --speech
```

Используйте версию, существование которой подтверждено в GHCR. Для строгой
неизменности замените ссылки на `ghcr.io/r1vendev/transcrisummaryzator@sha256:...`
из отчёта публикации: registry tags технически изменяемы, digest фиксирует
содержимое. GitHub описывает [скачивание по digest](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

Перед обновлением рабочих данных сделайте остановленный backup. Не используйте
`docker compose down -v`: флаг `-v` удаляет тома с записями и ключом шифрования.

### Backup и восстановление

Пример для default summary-проекта. Каталог backup не храните в Git и защитите
на уровне диска/backup-системы; архив секретов храните отдельно от архива данных.

```sh
umask 077
mkdir -p "$HOME/transcri-backup"
docker compose stop
docker compose run --rm -T --no-deps --entrypoint tar app -C /data -czf - . > "$HOME/transcri-backup/data.tar.gz"
docker compose run --rm -T --no-deps --entrypoint tar app -C /secrets -czf - . > "$HOME/transcri-backup/secrets.tar.gz"
docker compose start
```

Для новой установки создайте пустые тома Compose и восстановите оба архива
**до bootstrap**. В каталоге checkout после сборки образа:

```sh
docker compose run --rm -T --no-deps --entrypoint tar app -C /data -xzf - < "$HOME/transcri-backup/data.tar.gz"
docker compose run --rm -T --no-deps --entrypoint tar bootstrap -C /secrets -xzf - < "$HOME/transcri-backup/secrets.tar.gz"
./scripts/docker-install.sh --summary
```

Восстанавливайте в пустые тома после остановки всех их пользователей. Ключ
шифрования должен соответствовать базе. Потерянный master key нельзя заменить
новым без потери доступа к сохранённым токенам. Bootstrap обнаруживает
сохранённую базу ключей без master key и останавливается.

### Пароль администратора

Если пароль потерян, на доверенном Docker-хосте выполните:

```sh
docker compose run --rm --no-deps bootstrap reset-admin
docker compose restart app scheduler
```

Новый пароль показывается один раз; ключ шифрования и сохранённые API-токены
остаются прежними. Команда требует доступа к Docker-хосту.

## Проверка и диагностика

```sh
docker compose ps
docker compose logs --tail=100 app scheduler
docker compose exec app python -m unittest discover -s docker -p 'test_*.py'
```

Healthcheck проверяет только доступность HTTP API. Состояние `healthy` не
подтверждает доступ к платной модели или качество саммари. В речевом профиле
дополнительно выполните `docker compose -f compose.yaml -f compose.speech.yaml
exec app python pipeline.py doctor`, затем примите контрольную запись на GPU.
`doctor` проверяет наличие исполняемых файлов, а не качество распознавания.

### Проверено для этой поставки

- Лёгкий Linux amd64 образ собран; размер около 632 МБ. Оба Compose-файла
  проходят `docker compose config`.
- Пять offline-тестов контейнера проверили bootstrap, сохранение config/ключей
  и паузы очереди, смену пароля без смены master key, отказ при потерянном ключе
  и безопасный импорт. Два теста публикации проверили повторное использование
  SHA-образа и отказ при конфликте release-тега до записи новых aliases.
- Реальный Docker smoke проверил HTTP API, 401 без администратора, 200 после
  входа, 403 для неверного Host, импорт синтетической стенограммы и сохранение
  записи/пароля после restart. Саммари осталось `not_started`: платных запросов
  не было. Smoke удалил только собственные временные контейнеры и тома.
- Полная Linux amd64 речевая сборка завершилась с exit 0. Согласованность
  зависимостей и импорты GigaAM, Silero, Faster Whisper, DiariZen и NeMo
  проверены без сети. Docker archive — 13 508 044 288 байт. Эта сборка подтвердила
  окружения; публикация финального кода выполняется отдельным workflow.
- Скачивание весов, inference и production cutover не выполнялись.
  Модельные параметры работающего сервера не изменялись.

Политика build context исключает `.env`, локальные config, медиа, базы, venv,
кэш и файлы секретов через `.dockerignore`. Docker описывает эту границу в
[документации build context](https://docs.docker.com/build/concepts/context/).
