# real_estate_mlops

MLOps-проект для прогноза цены за квадратный метр недвижимости с трекингом
в MLflow и контейнеризованным FastAPI-сервисом инференса.

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

Сервис: `http://localhost:8000`

## CI-пайплайн

Workflow GitHub Actions: `.github/workflows/ci.yml`

Этапы пайплайна:

1. Установка зависимостей из `requirements-dev.txt`
2. Линт (`flake8 src/api tests`)
3. Тесты (`pytest -q`)
4. Сборка Docker-образа
5. Smoke-проверка Docker (`GET /health`)
