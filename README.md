# real_estate_mlops

MLOps-проект для прогноза цены за квадратный метр недвижимости с трекингом
в MLflow и контейнеризованным FastAPI-сервисом инференса.

## Архитектура проекта

Проект опирается на шаблон `Cookiecutter Data Science`, но дополнен
MLOps-компонентами: `DVC`, `MLflow`, `FastAPI`, `Docker`,
`Prometheus`, `Grafana` и `GitHub Actions`.

```text
real_estate_mlops
├── .github/
│   └── workflows/
│       ├── ci.yml                  <- CI: линт, тесты, сборка Docker, smoke-check
│       └── cd.yml                  <- CD: публикация образа и деплой на сервер
├── .dvc/
│   └── config                      <- Настройки DVC remote
├── data/
│   ├── processed/                  <- Подготовленные датасеты для обучения
│   └── raw/                        <- Сырые данные, отслеживаемые через DVC
├── docs/                           <- Sphinx-документация
├── mlartifacts/                    <- Локальные артефакты MLflow
├── monitoring/
│   ├── grafana/                    <- Provisioning и dashboard для Grafana
│   └── prometheus/                 <- Конфигурация сбора метрик Prometheus
├── models/
│   ├── model/                      <- Production-модель для API
│   └── model_grad_boost/           <- Дополнительные артефакты экспериментов
├── reports/
│   ├── registry_result.json        <- Результат регистрации модели в MLflow Registry
│   └── train_metrics.json          <- Метрики baseline-модели
├── src/
│   ├── api/
│   │   ├── main.py                 <- FastAPI-приложение и OpenAPI
│   │   ├── model_loader.py         <- Загрузка production-модели
│   │   └── schemas.py              <- Pydantic-схемы запросов и ответов
│   ├── data/
│   │   └── make_dataset.py         <- Подготовка датасета
│   └── models/
│       ├── register_mlflow_model.py <- Регистрация run в MLflow Registry
│       ├── sweep_models.py         <- Перебор baseline-кандидатов
│       ├── train_dvc_model.py      <- Обучение baseline через DVC + MLflow
│       └── train_model.py          <- Локальный сценарий обучения
├── tests/
│   └── test_api.py                 <- Smoke/API-тесты
├── .env.example                    <- Пример переменных окружения
├── docker-compose.yml              <- Локальный и серверный запуск контейнера
├── Dockerfile                      <- Сборка образа FastAPI-сервиса
├── dvc.yaml                        <- Описание DVC-пайплайна
├── params.yaml                     <- Параметры обучения и MLflow Registry
├── README.md                       <- Описание проекта и инструкции по запуску
├── requirements.txt                <- Основные зависимости
├── requirements-dev.txt            <- Зависимости для разработки и тестов
├── setup.py                        <- Установка пакета через pip install -e .
└── tox.ini                         <- Конфигурация flake8/tox
```

## Контракт модели для API

API инференса ожидает production-артефакты в `models/model`:

- `models/model/model.pkl` - сериализованная sklearn-совместимая модель.
- `models/model/metadata.json` - метаданные с полем `model_version`.

Пример схемы метаданных находится в `models/model/metadata.example.json`.

## Переменные окружения

Перед локальным запуском скопируйте значения по умолчанию:

```bash
cp .env.example .env
```

Альтернатива для PowerShell:

```powershell
Copy-Item .env.example .env
```

Поддерживаемые переменные:

- `MODEL_PATH` (по умолчанию: `models/model/model.pkl`)
- `MODEL_METADATA_PATH` (по умолчанию: `models/model/metadata.json`)

## DVC-пайплайн

В проекте настроен воспроизводимый пайплайн `dvc.yaml`:

1. `prepare` — подготовка датасета:
   - вход: `data/raw/russia_real_estate.csv`
   - выход: `data/processed/cleaned_data.csv`
2. `train_baseline` — обучение baseline-модели:
   - вход: `data/processed/cleaned_data.csv`
   - выход: `models/model/model.pkl`
   - метаданные: `models/model/metadata.json`
   - метрики: `reports/train_metrics.json`

Параметры обучения задаются в `params.yaml`.

Запуск:

```bash
dvc repro
```

Проверка состояния:

```bash
dvc status
dvc metrics show
```

Настроенный DVC remote (локальный путь рядом с репозиторием):

```bash
dvc remote list
```

## Связка с MLflow Registry

После обучения baseline DVC запускает стадию `register_model`, которая:

1. Берёт `run_id` из `models/model/metadata.json`.
2. Регистрирует версию модели в MLflow Registry.
3. Назначает alias `champion`.
4. Переводит версию в стадию `Staging`.
5. Сохраняет результат в `reports/registry_result.json`.

Параметры Registry настраиваются в `params.yaml`:

- `mlflow.registry.model_name`
- `mlflow.registry.alias`
- `mlflow.registry.stage`

## Локальный запуск Python (без Docker)

```bash
python -m pip install -r requirements-dev.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI: `http://localhost:8000/docs`

## Запуск в Docker

Сборка образа:

```bash
docker build -t real-estate-api:local .
```

Запуск контейнера:

```bash
docker run --rm -p 8000:8000 --env-file .env real-estate-api:local
```

## Запуск через Docker Compose

```bash
docker compose up --build
```

Сервисы после запуска:

- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin/admin`)

Для деплоя через registry можно переопределить образ:

```bash
API_IMAGE=ghcr.io/<owner>/real-estate-mlops-api:latest docker compose up -d
```

## Мониторинг

В проект добавлен базовый стек мониторинга:

- `Prometheus` собирает метрики с endpoint `GET /metrics`
- `Grafana` автоматически подключается к `Prometheus`
- dashboard `Real Estate API Overview` загружается при старте Grafana

API публикует следующие ключевые метрики:

- `real_estate_api_requests_total` - количество HTTP-запросов по path/status
- `real_estate_api_request_latency_seconds` - latency запросов
- `real_estate_api_predictions_total` - число запросов инференса
- `real_estate_api_model_ready` - готовность модели к предсказаниям

### Примеры запросов в Prometheus

Интерфейс Prometheus доступен по адресу `http://localhost:9090`.
Ниже приведены готовые PromQL-запросы для ручной проверки сервиса.

Готовность модели:

```promql
real_estate_api_model_ready
```

Общее число HTTP-запросов:

```promql
real_estate_api_requests_total
```

Скорость входящих запросов за последние 5 минут:

```promql
sum by (path) (rate(real_estate_api_requests_total[5m]))
```

Количество запросов инференса за последние 15 минут:

```promql
sum by (status) (increase(real_estate_api_predictions_total[15m]))
```

Количество ошибок API по статус-кодам:

```promql
sum by (status_code, path) (increase(real_estate_api_requests_total{status_code=~"4..|5.."}[15m]))
```

95-й перцентиль latency по endpoint:

```promql
histogram_quantile(
  0.95,
  sum by (le, path) (rate(real_estate_api_request_latency_seconds_bucket[5m]))
)
```

## CI-пайплайн

Workflow GitHub Actions: `.github/workflows/ci.yml`

Этапы пайплайна:

1. Установка зависимостей из `requirements-dev.txt`
2. Линт (`flake8 src/api tests`)
3. Тесты (`pytest -q`)
4. Сборка Docker-образа
5. Smoke-проверка Docker (`GET /health`)

## CD-пайплайн

Workflow GitHub Actions: `.github/workflows/cd.yml`

Что делает CD после успешного `CI` для `main`:

1. Собирает Docker-образ приложения.
2. Публикует образ в `GHCR` с тегами `<commit_sha>` и `latest`.
3. Подключается к серверу по `SSH`.
4. Обновляет репозиторий на сервере до `main`.
5. Выполняет `docker compose pull` и `docker compose up -d` с новым тегом образа.

### Что должно быть на сервере

- установлен Docker с Compose plugin
- установлен Git
- репозиторий уже склонирован в директорию деплоя
- рядом с `docker-compose.yml` создан файл `.env`

### Secrets для GitHub Actions

Для работы CD нужно добавить secrets в репозиторий:

- `DEPLOY_HOST` - адрес сервера
- `DEPLOY_USER` - пользователь для SSH
- `DEPLOY_SSH_KEY` - приватный SSH-ключ
- `DEPLOY_PATH` - путь к проекту на сервере
- `GHCR_USERNAME` - пользователь/аккаунт с доступом на чтение пакетов GHCR
- `GHCR_TOKEN` - token с правом `read:packages`

После merge в `main` деплой запускается автоматически.
