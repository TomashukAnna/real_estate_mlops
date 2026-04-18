# Контракт production-модели

Контракт для инференса модели предсказания.

Обязательно нужно:
- `model.pkl` - файл модели.
- `metadata.json` - метаданные модели `model_version`.

Переменные окружения
- `MODEL_PATH`
- `MODEL_METADATA_PATH`

Определить надо в файле окружения `.env`:
- `MODEL_PATH=models/model/model.pkl`
- `MODEL_METADATA_PATH=models/model/metadata.json`
