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
            
            cls._instance = AppConfig(**raw_data)
        
        return cls._instance

# Global accessor
settings = ResonaConfig.load()