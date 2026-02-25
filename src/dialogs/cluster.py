from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.models import DialogProject, LineStatus, DialogLine

@dataclass
class DialogCluster:
    """
    Represents a bounded batch of dialog lines sharing identical generation parameters.

    Attributes:
        voice_id (str): The identifier of the voice profile to use.
        language (str): The language code for the generation.
        emotion_id (Optional[str]): The canonical identifier for the emotion, if any.
        intensity_id (Optional[str]): The canonical identifier for the emotion intensity, if any.
        texts (List[str]): The ordered list of text strings to synthesize.
        state_ids (List[str]): The database identifiers of the line states corresponding to the texts.
    """
    voice_id: str
    language: str
    emotion_id: Optional[str]
    intensity_id: Optional[str]
    texts: List[str]
    state_ids: List[str]

class DialogClusterer:
    """
    Service responsible for aggregating pending dialog lines into optimized, 
    memory-safe batches.
    """

    def build_clusters(self, project: DialogProject, max_batch_size: int = 8) -> List[DialogCluster]:
        """
        Groups pending line states into homogenous clusters using precise database IDs,
        enforcing a maximum batch size to prevent GPU out-of-memory errors.

        Args:
            project (DialogProject): The dialog project containing the states and definitions.
            max_batch_size (int): The maximum number of lines a single cluster can hold.

        Returns:
            List[DialogCluster]: A list of clusters ready to be processed sequentially.
        """
        clusters: List[DialogCluster] = []
        active_groups: Dict[Tuple[str, str, Optional[str], Optional[str]], DialogCluster] = {}
        
        line_map: Dict[str, DialogLine] = {line.id: line for line in project.definition.script}
        sorted_states = sorted(project.states, key=lambda s: s.index)

        for state in sorted_states:
            if state.status != LineStatus.PENDING:
                continue
                
            line = line_map.get(state.line_id)
            if not line:
                continue

            lang = line.language or project.definition.default_language
            
            key = (
                line.voice_id,
                lang,
                state.emotion_id,
                state.intensity_id
            )

            if key not in active_groups:
                active_groups[key] = DialogCluster(
                    voice_id=line.voice_id,
                    language=lang,
                    emotion_id=state.emotion_id,
                    intensity_id=state.intensity_id,
                    texts=[],
                    state_ids=[]
                )

            current_cluster = active_groups[key]
            current_cluster.texts.append(line.text)
            current_cluster.state_ids.append(state.id)

            if len(current_cluster.texts) >= max_batch_size:
                clusters.append(current_cluster)
                del active_groups[key]

        clusters.extend(active_groups.values())

        return clusters