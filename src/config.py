import os
import yaml
from pathlib import Path
from src.models import AppConfig

class ResonaConfig:
    _instance: AppConfig = None

    @classmethod
    def load(cls) -> AppConfig:
        if cls._instance is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                raw_data = yaml.safe_load(f)
            
            config = AppConfig(**raw_data)
                        
            db_dir = Path(config.paths.db_file).parent
            os.makedirs(db_dir, exist_ok=True)
            
            os.makedirs(config.paths.assets_dir, exist_ok=True)
            os.makedirs(config.paths.temp_dir, exist_ok=True)
            os.makedirs(config.paths.tunespace_dir, exist_ok=True)
            os.makedirs(config.paths.emotionspace_dir, exist_ok=True)
            
            api_temp_path = Path(config.paths.apispace_dir) / "temp"
            os.makedirs(api_temp_path, exist_ok=True)
            
            cls._instance = config
        
        return cls._instance

settings = ResonaConfig.load()