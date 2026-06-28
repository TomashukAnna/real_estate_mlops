# real_estate_mlops

MLOps-проект для прогноза цены за квадратный метр недвижимости с трекингом
в MLflow и контейнеризованным FastAPI-сервисом инференса.

## Архитектура проекта

Проект опирается на шаблон `Cookiecutter Data Science`, но дополнен
MLOps-компонентами: `DVC`, `MLflow`, `FastAPI`, `Docker`,
`Prometheus`, `Grafana` и `GitHub Actions`.

Дополнительно в стек включены два контейнера интерфейсного слоя:

- `bff` - backend-for-frontend для UI, который работает с датасетом,
  вызывает inference API, запускает переобучение и агрегирует данные
  из MLflow;
- `ui` - Streamlit-приложение с картой, таблицей предсказаний,
  anomaly flags и страницей экспериментов.

```text
real_estate_mlops
├── .github/
│   └── workflows/
│       ├── ci.yml                  <- CI: отдельные jobs для линта, тестов, сборки и smoke-check
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
├── k8s/                            <- Kubernetes-манифесты для Docker Desktop Kubernetes и Argo CD
├── argocd/                         <- Argo CD Application-манифесты
├── ui/                             <- Streamlit UI с картой и страницей экспериментов
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
│   ├── bff/
│   │   ├── main.py                 <- BFF API для UI
│   │   ├── dataset_service.py      <- Точки карты и последние предсказания
│   │   ├── mlflow_service.py       <- Сводка по experiment runs и registry
│   │   └── retrain_service.py      <- Запуск обучения и reload модели
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
- `models/model/metadata.json` - метаданные с полями `model_version` и `reference_profile` для drift-мониторинга.

Пример схемы метаданных находится в `models/model/metadata.example.json`.

## Переменные окружения

Перед локальным запуском скопируйте значения по умолчанию:

```bash
cp .env.example .env
```

Поддерживаемые переменные:

- `MODEL_PATH` (по умолчанию: `models/model/model.pkl`)
- `MODEL_METADATA_PATH` (по умолчанию: `models/model/metadata.json`)
- `INFERENCE_LOG_PATH` (по умолчанию: `reports/inference/predictions.jsonl`)
- `DRIFT_REPORT_PATH` (по умолчанию: `reports/drift/latest_drift_report.json`)
- `DRIFT_HTML_PATH` (по умолчанию: `reports/drift/latest_drift_report.html`)
- `DRIFT_WINDOW_SIZE` (по умолчанию: `500`)
- `DRIFT_INTERVAL_SECONDS` (по умолчанию: `60`)
- `DRIFT_METRICS_PORT` (по умолчанию: `8001`)
- `MLFLOW_TRACKING_URI` (по умолчанию: `http://host.docker.internal:5000`)
- `BFF_DATASET_PATH` (по умолчанию: `data/processed/cleaned_data.csv`)

## DVC-пайплайн

В проекте настроен воспроизводимый пайплайн `dvc.yaml`:

1. `prepare` — подготовка датасета:
   - вход: `data/raw/russia_real_estate.csv`
   - выход: `data/processed/cleaned_data.csv`
2. `train_baseline` — обучение baseline-модели:
   - вход: `data/processed/cleaned_data.csv`
   - выход: `models/model/model.pkl`
   - метаданные: `models/model/metadata.json` с `reference_profile`
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

Важно для Git-репозитория:

- локальный каталог DVC remote `real_estate_mlops_dvc_remote/` не коммитится в Git;
- каталог `data/raw/` заполняется после `dvc pull`, в Git хранится только `.dvc`-файл
  `data/raw/russia_real_estate.csv.dvc`.

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
- BFF API: `http://localhost:8002`
- UI: `http://localhost:8080`
- Drift metrics: `http://localhost:8001/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin/admin`)

### MLflow на хосте и контейнер `bff`

Контейнер `bff` обращается к MLflow по `MLFLOW_TRACKING_URI`, обычно
`http://host.docker.internal:5000`. Начиная с **MLflow 3.5+**, сервер проверяет
заголовок `Host` и по умолчанию может отвечать **403** с текстом вроде
`Invalid Host header - possible DNS rebinding attack detected`.

Запускайте трекинг-сервер с разрешёнными именами (порт в записи хоста часто
нужен для клиентов с `Host: …:5000`):

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "host.docker.internal:5000,host.docker.internal,localhost:*,127.0.0.1:*"
```

Только для локальной отладки (не для продакшена): `--disable-security-middleware`.

### Что умеет UI

UI на `http://localhost:8080` использует `bff` и предоставляет:

- страницу инференса с картой объектов недвижимости;
- кнопку `Добавить новую квартиру`, которая выбирает случайную unseen-точку
  из `data/processed/cleaned_data.csv`, отправляет признаки в `/predict` и
  окрашивает маркер в красный/зелёный по сравнению с фактической ценой;
- popup по клику на маркер со всеми признаками квартиры;
- таблицу последних предсказаний и anomaly flags;
- кнопку `Запустить переобучение`, которая запускает baseline-обучение,
  регистрацию в MLflow Registry и `POST /reload-model` для inference API;
- страницу экспериментов с последними run-ами из MLflow.

Начиная с текущей версии UI работает на `Streamlit`, а не на SPA-сборке.

## Запуск в Docker Desktop Kubernetes

Kubernetes-манифесты организованы вокруг уже существующих Dockerfile:

- API собирается из `Dockerfile`;
- BFF собирается из `Dockerfile.bff`;
- UI собирается из `ui/Dockerfile`;
- drift-сервис использует тот же API-образ и запускает `python -m src.monitoring.drift_service`.

Структура манифестов:

- `k8s/base` содержит приложение: общий config, локальные тома и `api`/`bff`/`ui`/`drift`;
- `k8s/overlays/monitoring` добавляет Prometheus и Grafana;
- `k8s/overlays/docker-desktop` собирает локальный стек целиком;
- `k8s/overlays/argocd` использует ту же базу, но подменяет локальные образы на GHCR.

Перед применением манифестов включите Kubernetes в Docker Desktop:
`Settings -> Kubernetes -> Enable Kubernetes`.

Быстрый запуск одной командой:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-k8s.ps1
```

Скрипт собирает локальные Docker-образы, очищает namespace `real-estate-mlops`
и локальные PV, применяет `k8s/overlays/docker-desktop`, ждёт готовности
deployment-ов и запускает `port-forward` для API, BFF, UI, Prometheus и Grafana.
Оставьте окно скрипта открытым, пока нужны локальные URL; при `Ctrl+C` или
закрытии окна скрипт остановит запущенные port-forward процессы.

Если образы уже собраны, можно пропустить Docker build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-k8s.ps1 -UseExistingImages
```

Если нужен ручной запуск без скрипта, выполните шаги ниже.

Соберите локальные образы:

```powershell
docker build -t real-estate-api:local .
docker build -t real-estate-bff:local -f Dockerfile.bff .
docker build -t real-estate-ui:local ./ui
```

Манифесты монтируют локальные директории `data`, `models` и `reports`
через `hostPath`. Если проект лежит в другой директории, обновите пути в
`k8s/base/storage.yaml`.

Примените локальный overlay:

```powershell
kubectl apply -k k8s/overlays/docker-desktop
kubectl get pods -n real-estate-mlops
kubectl get svc -n real-estate-mlops
```

Открыть сервисы локально можно через `port-forward`:

```powershell
kubectl port-forward -n real-estate-mlops svc/real-estate-api 8000:8000
kubectl port-forward -n real-estate-mlops svc/real-estate-bff 8002:8002
kubectl port-forward -n real-estate-mlops svc/real-estate-ui 8080:8501
kubectl port-forward -n real-estate-mlops svc/real-estate-prometheus 9090:9090
kubectl port-forward -n real-estate-mlops svc/real-estate-grafana 3000:3000
```

После этого доступны:

- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`
- BFF API: `http://localhost:8002`
- UI: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin/admin`)

## Argo CD

Для локального GitOps-деплоя в Docker Desktop Kubernetes установите Argo CD:

```powershell
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=available --timeout=180s deployment/argocd-server -n argocd
kubectl port-forward svc/argocd-server -n argocd 8081:443
```

Пароль администратора:

```powershell
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
```

UI Argo CD: `https://localhost:8081`, логин `admin`.

Локальный Application для Docker Desktop Kubernetes:

```powershell
kubectl apply -f argocd/application-docker-desktop.yaml
```

Этот Application синхронизирует `k8s/overlays/docker-desktop` и ожидает локально
собранные образы `real-estate-api:local`, `real-estate-bff:local`,
`real-estate-ui:local`.

Для CD через GHCR используется `argocd/application-ghcr.yaml` и overlay
`k8s/overlays/argocd`. Он указывает на образы:

- `ghcr.io/tomashukanna/real-estate-mlops-api:latest`
- `ghcr.io/tomashukanna/real-estate-mlops-bff:latest`
- `ghcr.io/tomashukanna/real-estate-mlops-ui:latest`

Если основная ветка будет переименована с `master` на `main`, обновите
`targetRevision` в `argocd/*.yaml`.

### Контракт `/predict`

`POST /predict` принимает все модельные признаки и необязательное поле
`actual_price_per_m2`.

- Если `actual_price_per_m2 = -1`, наблюдение считается unlabeled и не участвует в target/concept drift.
- Если передано положительное значение, оно попадает в inference log и участвует в расчётах drift.

Все успешные предсказания сохраняются в `reports/inference/predictions.jsonl`.

Для деплоя через registry можно переопределить образ:

```bash
API_IMAGE=ghcr.io/<owner>/real-estate-mlops-api:latest docker compose up -d
```

## Мониторинг

В проект добавлен базовый стек мониторинга:

- `Prometheus` собирает метрики с endpoint `GET /metrics`
- отдельный контейнер `drift` рассчитывает `data drift`, `prediction drift`, `target drift`, `concept drift`
- `Grafana` автоматически подключается к `Prometheus`
- `Loki + Promtail` собирают логи контейнеров и показывают их в Grafana
- dashboard `Real Estate API Overview` загружается при старте Grafana

API публикует следующие ключевые метрики:

- `real_estate_api_requests_total` - количество HTTP-запросов по path/status
- `real_estate_api_request_latency_seconds` - latency запросов
- `real_estate_api_predictions_total` - число запросов инференса
- `real_estate_api_labeled_predictions_total` - число запросов с фактической ценой
- `real_estate_api_model_ready` - готовность модели к предсказаниям

Drift-сервис публикует:

- `real_estate_drift_service_up` - статус drift-контейнера
- `real_estate_data_drift_score` - средний data drift score по признакам
- `real_estate_data_drifted_features_total` - число drifted features
- `real_estate_prediction_drift_score` - drift по распределению prediction
- `real_estate_target_drift_score` - drift по фактическим таргетам
- `real_estate_concept_drift_score` - деградация качества на labeled-окне
- `real_estate_concept_drift_mae`, `real_estate_concept_drift_rmse`, `real_estate_concept_drift_r2`

Дополнительно сервис генерирует:

- `reports/drift/latest_drift_report.json`
- `reports/drift/latest_drift_report.html`

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

Количество labeled-запросов за последние 15 минут:

```promql
sum by (model_version) (increase(real_estate_api_labeled_predictions_total[15m]))
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

Средний score data drift:

```promql
real_estate_data_drift_score
```

MAE на labeled-окне drift-сервиса:

```promql
real_estate_concept_drift_mae
```

## CI-пайплайн

Workflow GitHub Actions: `.github/workflows/ci.yml`

Этапы пайплайна выполняются раздельно, каждый в своём `GitHub Actions job`
(`ubuntu-latest` runner):

1. `lint` - установка зависимостей из `requirements-dev.txt` и запуск `flake8 src/api tests`
2. `test` - отдельная установка зависимостей и запуск `pytest -q`
3. `build` - отдельная сборка Docker-образа после успешных `lint` и `test`
4. `smoke` - отдельная smoke-проверка `GET /health` на образе, переданном из `build` через artifact

## CD-пайплайн

Workflow GitHub Actions: `.github/workflows/cd.yml`

Что делает CD после успешного `CI` для `main` или текущей `master`:

1. Собирает API-образ из `Dockerfile`.
2. Собирает BFF-образ из `Dockerfile.bff`.
3. Собирает UI-образ из `ui/Dockerfile`.
4. Публикует все образы в `GHCR` с тегами `<commit_sha>` и `latest`.
5. Обновляет `newTag` в `k8s/overlays/argocd/kustomization.yaml` на `<commit_sha>`.
6. Деплой в Kubernetes выполняет Argo CD, синхронизируя манифесты из Git.

### Что нужно для GitOps CD

- Kubernetes-кластер с установленным Argo CD.
- Argo CD Application из `argocd/application-ghcr.yaml`.
- Доступ к `GHCR` из кластера, если registry package приватный.
- Обновлённые `repoURL` и `targetRevision` в `argocd/*.yaml`, если репозиторий или основная ветка отличаются.

### Secrets для GitHub Actions

Для публикации образов используется стандартный `GITHUB_TOKEN` с правом
`packages: write`. Дополнительные SSH-secrets больше не требуются.

После merge в основную ветку GitHub Actions публикует образы и фиксирует новый
tag в Git, а Argo CD подтягивает актуальное состояние и синхронизирует
Kubernetes-приложение.
