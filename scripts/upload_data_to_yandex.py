"""
Скрипт для переноса данных на Яндекс Диск с перезаписью
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Загружаем .env файл
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.yandex_storage import YandexStorage


def main():
    print("🚀 Starting upload to Yandex Disk...")
    storage = YandexStorage()
    
    # Список файлов для загрузки: (локальный путь, удаленный путь)
    files_to_upload = [
        ("data/raw/russia_real_estate.csv", "data/raw/russia_real_estate.csv"),
        ("data/processed/cleaned_data.csv", "data/processed/cleaned_data.csv"),
        ("models/model/model.pkl", "models/model.pkl"),
        ("models/model/metadata.json", "metadata/metadata.json"),
    ]
    
    uploaded_count = 0
    skipped_count = 0
    
    for local_path, remote_path in files_to_upload:
        local = Path(local_path)
        if not local.exists():
            print(f"⚠️ Skipped (not found): {local_path}")
            skipped_count += 1
            continue
        
        # Проверяем, существует ли файл на диске
        full_remote = storage._full_path(remote_path)
        if storage.client.exists(full_remote):
            print(f"⚠️ File exists: {remote_path} — replacing...")
            storage.client.remove(full_remote)
        
        # Загружаем с перезаписью
        success = storage.upload_file(local, remote_path, overwrite=True)
        if success:
            uploaded_count += 1
            print(f"✅ Uploaded: {remote_path}")
        else:
            print(f"❌ Failed: {remote_path}")
    
    print(f"\n📊 Summary: {uploaded_count} uploaded, {skipped_count} skipped")
    
    # Проверяем, что файлы загружены
    print("\n🔍 Checking uploaded files...")
    for _, remote_path in files_to_upload:
        if storage.file_exists(remote_path):
            print(f"{remote_path} — exists")
        else:
            print(f"{remote_path} — NOT found")


if __name__ == "__main__":
    main()