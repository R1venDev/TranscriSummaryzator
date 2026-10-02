# Публикация Docker-образов в GitHub Container Registry

Workflow `.github/workflows/docker-publish.yml` собирает `summary` и `speech`
для `linux/amd64` из финального Git commit. Он запускается после push в `main`,
по тегу `v*` или вручную из Actions. Оба job имеют только `contents: read` и
`packages: write`; вход в GHCR выполняется краткоживущим `GITHUB_TOKEN`.
Personal access token в коде, build args или секретах репозитория не требуется.
См. [официальный процесс GitHub](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

## Теги и проверки

- `sha-<полный Git SHA>-summary` и `sha-<полный Git SHA>-speech` фиксируют сборку
  данного commit. Если тег уже существует, workflow использует его digest.
- `summary` и `speech` — обновляемые каналы `main`. Оба обновляются отдельным
  job только после успеха всей матрицы. Summary проходит Docker smoke с
  синтетическим импортом, проверкой авторизации и restart.
- Тег Git `v2026.10.02` или manual input `release_version=2026.10.02` создаёт
  `2026.10.02-summary` и `2026.10.02-speech`. Уже занятый release-тег с другим
  digest блокирует публикацию. Ручной запуск разрешён с `main` или release tag.
- BuildKit добавляет provenance и SBOM; фактические digest и опубликованные
  ссылки записываются в job summary. Это сведения о сборке, а не подтверждение
  качества распознавания или результатов аудита безопасности.

Публикации сериализованы, чтобы ручной повтор и push не заменяли одновременно
один тег SHA. Ошибка одного job блокирует promotion обоих каналов. Два registry
тега обновляются последовательно; установщик дополнительно проверяет, что
revision app и scheduler совпадают, и закрыто отказывает при несовпадении.
Не называйте релиз готовым до promotion и успешного скачивания обоих образов.

## Место на runner

Речевой job работает на отдельной одноразовой Ubuntu 24.04 машине GitHub.
Он удаляет предустановленные Android/GHC/.NET/toolcache каталоги этой машины и
проверяет наличие минимум 50 GiB свободного места. Этот шаг проверяет
`RUNNER_ENVIRONMENT=github-hosted` и не предназначен для собственного сервера.
При нехватке места job завершается до скачивания тяжёлых зависимостей.
Registry cache хранится под отдельными тегами `buildcache-summary` и
`buildcache-speech`; обычным пользователям они не нужны.

Dockerfile устанавливает зависимости до копирования приложения. Обновление
Python/HTML/JS поэтому сохраняет слои речевых окружений. В build context
попадают только разрешённые исходники и requirements; медиа, локальные config,
базы и секреты исключены. Последние import-проверки работают с отключённой
сетью. GPU и веса моделей не нужны для workflow.

## Первая публикация

1. Отправьте проверенный commit с workflow в `main` и дождитесь обоих job.
2. Проверьте страницу Packages репозитория и её visibility. Для установки без
   входа в GitHub оба варианта общего container package должны быть доступны
   публично. Права package связаны с репозиторием через OCI source label;
   проверьте фактическую видимость в интерфейсе GitHub.
3. В Actions запустите тот же workflow с `release_version=2026.10.02`, либо
   создайте Git tag `v2026.10.02` на этом commit. Существующие SHA-образы будут
   повторно использованы, новые version tags укажут на те же digest.
4. Проверьте `docker pull ghcr.io/r1vendev/transcrisummaryzator:speech` и
   аналогичную ссылку `:summary` без registry credentials, если package публичный.
5. Выполните установку по [DOCKER.md](DOCKER.md) на отдельном тестовом хосте.

Сами файлы workflow не доказывают успешную публикацию. Сохраните ссылки на
выполненные Actions jobs и manifest digest обоих образов в release notes.
