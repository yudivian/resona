from typing import Optional, Dict
from src.models import DialogProject, LineStatus, DialogLine
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager

class DialogResolver:
    """
    Service responsible for validating script resources and mapping 
    natural language labels to technical canonical identifiers.

    Attributes:
        voice_store (VoiceStore): The storage interface for validating voice profiles.
        emotion_manager (EmotionManager): The manager handling localized emotion catalogs.
    """

    def __init__(self, voice_store: VoiceStore, emotion_manager: EmotionManager):
        """
        Initializes the DialogResolver with required dependencies.

        Args:
            voice_store (VoiceStore): Storage instance to validate voice IDs.
            emotion_manager (EmotionManager): Manager instance to resolve emotion labels.
        """
        self.voice_store = voice_store
        self.emotion_manager = emotion_manager

    def resolve_project(self, project: DialogProject) -> None:
        """
        Validates and resolves all pending line states in a dialog project.

        This method mutates the provided project in-place. It iterates through
        the project states, looking up the corresponding creative line definition,
        and assigns canonical IDs for emotions and intensities. If a resource is
        missing or invalid, the state is marked as FAILED.

        Args:
            project (DialogProject): The dialog project containing states to resolve.
        """
        line_map: Dict[str, DialogLine] = {line.id: line for line in project.definition.script}

        for state in project.states:
            if state.status != LineStatus.PENDING:
                continue

            line = line_map.get(state.line_id)
            if not line:
                state.status = LineStatus.FAILED
                state.error = f"Orphaned state: DialogLine {state.line_id} not found."
                continue

            if not self.voice_store.get_profile(line.voice_id):
                state.status = LineStatus.FAILED
                state.error = f"Voice profile '{line.voice_id}' not found."
                continue

            lang = line.language or project.definition.default_language
            
            if line.emotion:
                eid = self._find_canonical_id(line.emotion, "emotions", lang)
                if eid:
                    state.emotion_id = eid
                else:
                    state.status = LineStatus.FAILED
                    state.error = f"Unresolved emotion label: {line.emotion}"
                    continue

            if line.intensity:
                iid = self._find_canonical_id(line.intensity, "intensifiers", lang)
                if iid:
                    state.intensity_id = iid
                else:
                    state.status = LineStatus.FAILED
                    state.error = f"Unresolved intensifier label: {line.intensity}"
                    continue

    def _find_canonical_id(self, label: str, category: str, lang: str) -> Optional[str]:
        """
        Performs a reverse lookup to find a technical ID from a localized string.

        Args:
            label (str): The natural language string provided by the user or LLM.
            category (str): The catalog category to search within (e.g., 'emotions').
            lang (str): The ISO language code to prioritize for the lookup.

        Returns:
            Optional[str]: The canonical ID if found, otherwise None.
        """
        catalog = self.emotion_manager.catalog.get(category, {})
        search_term = label.strip().lower()

        if search_term in catalog:
            return search_term

        for cid, data in catalog.items():
            names = data.get("localized_names", {})
            for code in [lang, "en"]:
                if names.get(code, "").strip().lower() == search_term:
                    return cid
        
        return None