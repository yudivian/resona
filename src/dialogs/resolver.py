from typing import Optional, Dict
from src.models import DialogProject, LineStatus, DialogLine
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager

class DialogResolver:
    """
    Service responsible for strictly validating script resources, ensuring data integrity 
    before generation, and mapping natural language labels to technical canonical identifiers.

    This class acts as a pre-flight validator for the synthesis pipeline. It ensures that 
    all requested acoustic resources (such as Voice Profiles) exist in the persistence layer, 
    and translates human-readable emotion and intensity labels into their respective 
    UUIDs or canonical keys defined within the system's global emotion catalog.
    """

    def __init__(self, voice_store: VoiceStore, emotion_manager: EmotionManager) -> None:
        """
        Instantiates the DialogResolver with its required external dependency managers.

        Args:
            voice_store (VoiceStore): The initialized database interface responsible for 
                                      storing and retrieving serialized VoiceProfile objects.
            emotion_manager (EmotionManager): The initialized handler containing the hierarchical 
                                              and localized catalog of emotional parameters.
        """
        self.voice_store = voice_store
        self.emotion_manager = emotion_manager

    def resolve_project(self, project: DialogProject) -> None:
        """
        Validates and resolves all pending line states within a specific dialog project.

        This method mutates the provided project instance in-place. It sequentially processes 
        every state whose status is marked as PENDING. For each state, it performs three 
        critical validation steps:
        1. Identifies if the requested voice profile ID actually exists in the local database.
        2. Resolves any natural language emotion labels to their underlying canonical identifiers.
        3. Resolves any natural language intensity modifiers to their underlying canonical identifiers.

        If any of these validations fail, the state's status is permanently altered to FAILED, 
        and an explicit error message is attached to prevent the engine from attempting to 
        process incomplete or non-existent tensor constraints.

        Args:
            project (DialogProject): The target data container holding both the creative 
                                     script definitions and their technical execution states.
        """
        line_map = {line.id: line for line in project.definition.script}

        for state in project.states:
            if state.status != LineStatus.PENDING:
                continue

            line = line_map.get(state.line_id)
            
            if not line:
                state.status = LineStatus.FAILED
                state.error = "Orphaned state: Script line definition could not be found."
                continue

            lang = line.language or project.definition.default_language

            voice_profile = self.voice_store.get_profile(line.voice_id)
            
            if not voice_profile:
                state.status = LineStatus.FAILED
                state.error = f"Unresolved voice profile: ID '{line.voice_id}' is missing from the database."
                continue

            if line.emotion:
                eid = self._find_canonical_id(line.emotion, "emotions", lang)
                if eid:
                    state.emotion_id = eid
                else:
                    state.status = LineStatus.FAILED
                    state.error = f"Unresolved emotion label: '{line.emotion}'"
                    continue

            if line.intensity:
                iid = self._find_canonical_id(line.intensity, "intensifiers", lang)
                if iid:
                    state.intensity_id = iid
                else:
                    state.status = LineStatus.FAILED
                    state.error = f"Unresolved intensifier label: '{line.intensity}'"
                    continue

    def _find_canonical_id(self, label: str, category: str, lang: str) -> Optional[str]:
        """
        Performs a reverse lookup operation to match a localized string against a technical ID.

        The method implements a fallback strategy: it first checks if the provided label is 
        already a canonical ID. If not, it searches the localized names associated with the 
        target language. If the target language fails, it gracefully degrades to searching 
        within the default English localized names.

        Args:
            label (str): The natural language string provided by the user or upstream systems.
            category (str): The top-level root key within the catalog to search (e.g., 'emotions').
            lang (str): The ISO 639-1 language code used to prioritize the string matching.

        Returns:
            Optional[str]: The strictly formatted canonical ID if a match is successfully identified, 
                           otherwise None.
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