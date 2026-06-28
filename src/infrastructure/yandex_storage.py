"""
Модуль для работы с Яндекс Диском.
Загрузка и выгрузка файлов, работа с DataFrame.
"""

import os
import tempfile
import json
import joblib
from pathlib import Path
from typing import Optional, Any, Dict
import pandas as pd

import yadisk


class YandexStorage:
    """Клиент для работы с Яндекс Диском"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("YANDEX_DISK_TOKEN")
        if not self.token:
            raise ValueError(
                "YANDEX_DISK_TOKEN is required. "
                "Set it in .env file or environment variables."
            )
        
        self.base_path = os.getenv("YANDEX_DISK_BASE_PATH", "/MLOps/real_estate")
        self.client = yadisk.YaDisk(token=self.token)
        
        # Проверяем подключение
        try:
            self.client.check_token()
            print(f"Yandex Disk connected. Base path: {self.base_path}")
        except Exception as e:
            print(f"Failed to connect to Yandex Disk: {e}")
    
    def _full_path(self, remote_path: str) -> str:
        """Полный путь к файлу на Яндекс Диске"""
        return f"{self.base_path}/{remote_path.lstrip('/')}"
    
    def upload_file(self, local_path: Path, remote_path: str, overwrite: bool = True) -> bool:
        """Загрузить файл на Яндекс Диск"""
        full_remote = self._full_path(remote_path)
        
        # Создаем папки, если их нет
        remote_dir = "/".join(full_remote.split("/")[:-1])
        if remote_dir and not self.client.exists(remote_dir):
            self.client.mkdir(remote_dir)
        
        try:
            # Добавляем параметр overwrite
            self.client.upload(str(local_path), full_remote, overwrite=overwrite)
            print(f"Uploaded: {local_path} → {full_remote}")
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Скачать файл с Яндекс Диска"""
        full_remote = self._full_path(remote_path)
        
        if not self.client.exists(full_remote):
            print(f"File not found: {full_remote}")
            return False
        
        try:
            self.client.download(full_remote, str(local_path))
            print(f"Downloaded: {full_remote} → {local_path}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def download_dataframe(self, remote_path: str) -> Optional[pd.DataFrame]:
        """Скачать CSV файл как pandas DataFrame"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            local_path = Path(tmp.name)
        
        if self.download_file(remote_path, local_path):
            try:
                df = pd.read_csv(local_path)
                return df
            finally:
                local_path.unlink()  # удаляем временный файл
        
        return None
    
    def upload_dataframe(self, df: pd.DataFrame, remote_path: str) -> bool:
        """Загрузить DataFrame как CSV на Яндекс Диск"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            local_path = Path(tmp.name)
            df.to_csv(local_path, index=False)
        
        try:
            return self.upload_file(local_path, remote_path)
        finally:
            local_path.unlink()
    
    def download_model(self, remote_path: str) -> Optional[Any]:
        """Скачать модель (pickle) с Яндекс Диска"""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            local_path = Path(tmp.name)
        
        if self.download_file(remote_path, local_path):
            try:
                model = joblib.load(local_path)
                return model
            finally:
                local_path.unlink()
        
        return None
    
    def upload_model(self, model: Any, remote_path: str) -> bool:
        """Загрузить модель (pickle) на Яндекс Диск"""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            local_path = Path(tmp.name)
            joblib.dump(model, local_path)
        
        try:
            return self.upload_file(local_path, remote_path)
        finally:
            local_path.unlink()
    
    def download_json(self, remote_path: str) -> Optional[Dict]:
        """Скачать JSON файл с Яндекс Диска"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            local_path = Path(tmp.name)
        
        if self.download_file(remote_path, local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            finally:
                local_path.unlink()
        
        return None
    
    def upload_json(self, data: Dict, remote_path: str) -> bool:
        """Загрузить JSON на Яндекс Диск"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            local_path = Path(tmp.name)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        try:
            return self.upload_file(local_path, remote_path)
        finally:
            local_path.unlink()
    
    def file_exists(self, remote_path: str) -> bool:
        """Проверить существование файла на Яндекс Диске"""
        return self.client.exists(self._full_path(remote_path))
    
    def list_files(self, remote_path: str = "") -> list:
        """Получить список файлов в папке"""
        full_remote = self._full_path(remote_path) if remote_path else self.base_path
        try:
            items = self.client.listdir(full_remote)
            return [item.name for item in items if item.type == "file"]
        except Exception as e:
            print(f"Failed to list files: {e}")
            return []