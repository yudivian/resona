import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.models import AppConfig

logger = logging.getLogger(__name__)

class EmotionManager:
    """
    Controller class responsible for managing the emotional prosody catalog.
    
    This manager resolves the core catalog as a project-local resource while
    orchestrating temporary assets within a dedicated workspace defined in 
    the central application configuration.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the EmotionManager with centralized path configurations.

        Args:
            config (AppConfig): The application configuration object.
        """
        self.config = config
        
        project_root = Path(__file__).parent.parent.parent
        self.catalog_path = project_root / "resources" / "emotions.json"
        
        self.workspace_dir = Path(config.paths.emotionspace_dir)
        self.temp_emotions_dir = self.workspace_dir / "temp"
        
        self.session_files: List[str] = []
        self.catalog: Dict[str, Any] = self._load_catalog()
        
        self._initialize_workspace()

    def _initialize_workspace(self) -> None:
        """
        Initializes the internal directory structure within the emotionspace.
        """
        try:
            os.makedirs(self.temp_emotions_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize emotionspace workspace: {e}")

    def _load_catalog(self) -> Dict[str, Any]:
        """
        Loads the prosody catalog from the project-local resource file.

        Returns:
            Dict[str, Any]: Parsed catalog data or default structure.
        """
        if not self.catalog_path.exists():
            logger.warning(f"Static catalog not found at {self.catalog_path}. Initializing empty.")
            return {"intensifiers": {}, "emotions": {}}
        
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error parsing local emotion catalog: {e}")
            return {"intensifiers": {}, "emotions": {}}

    def _save_catalog(self) -> bool:
        """
        Persists changes to the project-local resource catalog.

        Returns:
            bool: Success status of the save operation.
        """
        try:
            os.makedirs(self.catalog_path.parent, exist_ok=True)
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.catalog, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to persist changes to local catalog: {e}")
            return False

    def save_emotion(self, emotion_id: str, modifiers: Dict[str, float], localized_names: Dict[str, str]) -> bool:
        """
        Creates or updates an emotion entry in the catalog (Upsert).

        Args:
            emotion_id (str): Canonical English ID for the emotion.
            modifiers (Dict[str, float]): Dictionary containing 'temp', 'penalty', and 'top_p' deltas.
            localized_names (Dict[str, str]): Dictionary mapping language codes to translated names.

        Returns:
            bool: True if the operation was successful.
        """
        if "emotions" not in self.catalog:
            self.catalog["emotions"] = {}
            
        self.catalog["emotions"][emotion_id] = {
            "modifiers": modifiers,
            "localized_names": localized_names
        }
        return self._save_catalog()

    def update_emotion(self, emotion_id: str, modifiers: Optional[Dict[str, float]] = None, localized_names: Optional[Dict[str, str]] = None) -> bool:
        """
        Updates specific fields of an existing emotion entry.

        Args:
            emotion_id (str): The canonical ID of the emotion to update.
            modifiers (Optional[Dict[str, float]]): New deltas for generation parameters.
            localized_names (Optional[Dict[str, str]]): New translation mappings.

        Returns:
            bool: True if the update was applied, False if the ID was not found.
        """
        if emotion_id not in self.catalog.get("emotions", {}):
            return False
            
        if modifiers is not None:
            self.catalog["emotions"][emotion_id]["modifiers"] = modifiers
        if localized_names is not None:
            self.catalog["emotions"][emotion_id]["localized_names"] = localized_names
            
        return self._save_catalog()

    def delete_emotion(self, emotion_id: str) -> bool:
        """
        Removes an emotion entry from the catalog.

        Args:
            emotion_id (str): The canonical ID of the emotion to remove.

        Returns:
            bool: True if the deletion was successful.
        """
        if emotion_id in self.catalog.get("emotions", {}):
            del self.catalog["emotions"][emotion_id]
            return self._save_catalog()
        return False

    def save_intensifier(self, intensifier_id: str, multiplier: float, localized_names: Dict[str, str]) -> bool:
        """
        Creates or updates an intensifier entry in the catalog (Upsert).

        Args:
            intensifier_id (str): Canonical English ID for the intensifier.
            multiplier (float): The scaling multiplier applied to emotion deltas.
            localized_names (Dict[str, str]): Dictionary mapping language codes to translated names.

        Returns:
            bool: True if the operation was successful.
        """
        if "intensifiers" not in self.catalog:
            self.catalog["intensifiers"] = {}
            
        self.catalog["intensifiers"][intensifier_id] = {
            "multiplier": multiplier,
            "localized_names": localized_names
        }
        return self._save_catalog()

    def update_intensifier(self, intensifier_id: str, multiplier: Optional[float] = None, localized_names: Optional[Dict[str, str]] = None) -> bool:
        """
        Updates specific fields of an existing intensifier entry.

        Args:
            intensifier_id (str): The canonical ID of the intensifier to update.
            multiplier (Optional[float]): New scaling multiplier value.
            localized_names (Optional[Dict[str, str]]): New translation mappings.

        Returns:
            bool: True if the update was applied, False if the ID was not found.
        """
        if intensifier_id not in self.catalog.get("intensifiers", {}):
            return False
            
        if multiplier is not None:
            self.catalog["intensifiers"][intensifier_id]["multiplier"] = multiplier
        if localized_names is not None:
            self.catalog["intensifiers"][intensifier_id]["localized_names"] = localized_names
            
        return self._save_catalog()

    def delete_intensifier(self, intensifier_id: str) -> bool:
        """
        Removes an intensifier entry from the catalog.

        Args:
            intensifier_id (str): The canonical ID of the intensifier to remove.

        Returns:
            bool: True if the deletion was successful.
        """
        if intensifier_id in self.catalog.get("intensifiers", {}):
            del self.catalog["intensifiers"][intensifier_id]
            return self._save_catalog()
        return False

    def register_session_audio(self, file_path: str) -> None:
        """
        Tracks audio assets generated during the current application lifecycle.

        Args:
            file_path (str): The absolute path to the generated testing file.
        """
        if file_path not in self.session_files:
            self.session_files.append(file_path)

    def calculate_parameters(self, emotion_id: Optional[str] = None, intensifier_id: Optional[str] = None, 
                             emotion_override: Optional[Dict[str, float]] = None, 
                             intensifier_override: Optional[float] = None) -> Dict[str, float]:
        """
        Computes final generation hyperparameters by scaling emotional deltas.
        Supports live overrides for real-time UI previewing before saving.
        
        Args:
            emotion_id (Optional[str]): Canonical ID of the saved emotion.
            intensifier_id (Optional[str]): Canonical ID of the saved intensifier.
            emotion_override (Optional[Dict[str, float]]): Unsaved deltas for live preview.
            intensifier_override (Optional[float]): Unsaved multiplier for live preview.

        Returns:
            Dict[str, float]: Clamped values safe for the inference engine.
        """
        base_temp = 0.9
        base_penalty = 1.05
        base_top_p = 1.0
        
        mods = {"temp": 0.0, "penalty": 0.0, "top_p": 0.0}
        if emotion_override is not None:
            mods = emotion_override
        elif emotion_id:
            emotion = self.catalog.get("emotions", {}).get(emotion_id)
            if emotion:
                mods = emotion.get("modifiers", {})
        
        multiplier = 1.0
        if intensifier_override is not None:
            multiplier = intensifier_override
        elif intensifier_id:
            intensifier = self.catalog.get("intensifiers", {}).get(intensifier_id)
            if intensifier:
                multiplier = intensifier.get("multiplier", 1.0)
        
        calc_temp = base_temp + (mods.get("temp", 0.0) * multiplier)
        calc_penalty = base_penalty + (mods.get("penalty", 0.0) * multiplier)
        calc_top_p = base_top_p + (mods.get("top_p", 0.0) * multiplier)
        
        return {
            "temp": max(0.01, min(1.99, calc_temp)),
            "penalty": max(0.0, calc_penalty),
            "top_p": max(0.01, min(1.0, calc_top_p))
        }

    def get_localized_vocab(self, lang: str = "en") -> Dict[str, Dict[str, str]]:
        """
        Provides a mapping of {Canonical_ID: Display_Name} for the UI.
        
        If a translation is missing or empty for the requested language, 
        the Canonical ID is formatted and used as a fallback to ensure 
        the UI never renders an empty selection.
        """
        vocab = {"emotions": {}, "intensifiers": {}}
        
        for eid, data in self.catalog.get("emotions", {}).items():
            names = data.get("localized_names", {})
            # Logical fallback: Translation -> English Name -> Formatted ID
            val = names.get(lang) or names.get("en") or eid.replace("_", " ").capitalize()
            vocab["emotions"][eid] = val.strip() if val.strip() else eid.capitalize()
            
        for iid, data in self.catalog.get("intensifiers", {}).items():
            names = data.get("localized_names", {})
            val = names.get(lang) or names.get("en") or iid.replace("_", " ").capitalize()
            vocab["intensifiers"][iid] = val.strip() if val.strip() else iid.capitalize()
            
        return vocab

    def clear_session_audios(self) -> int:
        """
        Removes audio files tracked during the current runtime session.

        Returns:
            int: Number of files successfully deleted.
        """
        count = 0
        for file_path in self.session_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to remove session audio {file_path}: {e}")
        self.session_files.clear()
        return count

    def clear_all_temp_audios(self) -> int:
        """
        Performs a complete cleanup of the dedicated temp directory in emotionspace.

        Returns:
            int: Number of files successfully deleted.
        """
        count = 0
        if not self.temp_emotions_dir.exists():
            return count
        for filename in os.listdir(self.temp_emotions_dir):
            file_path = self.temp_emotions_dir / filename
            try:
                if file_path.is_file() and filename.endswith(".wav"):
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to remove temp file {file_path}: {e}")
        self.session_files.clear()
        return count
    
    def validate_entry(self, entry_id: str, entry_type: str, localized_names: Dict[str, str]) -> Tuple[bool, str]:
        """
        Validates the integrity of a new or updated catalog entry.

        Args:
            entry_id (str): The proposed canonical identifier.
            entry_type (str): Either 'emotions' or 'intensifiers'.
            localized_names (Dict[str, str]): Proposed translations.

        Returns:
            Tuple[bool, str]: Success status and an error message if validation fails.
        """
        clean_id = entry_id.strip().lower().replace(" ", "_")
        
        if not clean_id:
            return False, "Canonical ID cannot be empty."

        other_type = "intensifiers" if entry_type == "emotions" else "emotions"
        if clean_id in self.catalog.get(other_type, {}):
            return False, f"The ID '{clean_id}' is already used as an {other_type[:-1]}."

        existing_names = []
        for eid, data in self.catalog.get(entry_type, {}).items():
            if eid != clean_id:
                existing_names.extend([name.lower() for name in data.get("localized_names", {}).values()])

        for lang, name in localized_names.items():
            if name.lower() in existing_names:
                return False, f"The name '{name}' is already used in another {entry_type[:-1]}."

        return True, ""