import os
import io
import logging
import uuid
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi import BackgroundTasks 
from pydantic import ValidationError

from src.models import (
    VoiceSummary,
    VoiceDetail,
    VoiceEndpoints,
    EmotionSummary,
    IntensifierSummary,
    TaskStatusResponse,
    SynthesisRequest, 
    EmotionSynthesisRequest,
    DialogScript,
    DialogProject,
    ProjectSource,
    ProjectStatus,
    LineState,
    LineStatus,
    DialogGenerationResponse,
    DialogStatusResponse,
    DialogOutputs,
    DialogProgress,
    LineStatusTracker,
    LineIdentifier,
    DialogMergeResponse,
    ProjectListItem,
    ProjectListResponse,
)

from src.backend.store import VoiceStore
from src.backend.io import VoiceBundleIO
from src.api.dependencies import (
    get_store,
    get_embed_engine,
    get_config,
    get_emotion_manager,
    get_tts_provider,
    get_orchestrator
)
from src.emotions.manager import EmotionManager
from src.backend.engine import InferenceEngine


logger = logging.getLogger(__name__)
router = APIRouter()

ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}

def _process_synthesis_task(
    task_id: str,
    voice_id: str,
    text: str,
    lang: str,
    params: Optional[Dict[str, float]],
    store: VoiceStore,
    config: Any,
    provider: Any
):
    """
    Background worker that performs the actual neural inference.
    Updates the global ACTIVE_TASKS dictionary with the results.
    """
    try:
        profile = store.get_profile(voice_id)
        if not profile:
            raise Exception(f"Voice profile {voice_id} not found.")

        engine = InferenceEngine(config, provider, lang=lang)
        anchor = os.path.join(config.paths.assets_dir, profile.anchor_audio_path) if profile.anchor_audio_path else None
        engine.load_identity_from_state(profile.identity_embedding, profile.seed, anchor)

        output_name = f"api_{task_id}.wav"
        output_path = os.path.join(config.paths.emotionspace_dir, "temp", output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if params:
            final_path = engine.render_with_emotion(text, params, output_path)
        else:
            final_path = engine.render(text, output_path)

        ACTIVE_TASKS[task_id].update({
            "status": "completed",
            "file_path": final_path
        })
        logger.info(f"✅ Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}")
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "error": str(e)
        })


@router.get("/voices", response_model=List[VoiceSummary])
def list_voices(
    lang: Optional[str] = None,
    tag: Optional[str] = None,
    source_type: Optional[str] = None,
    limit: int = Query(
        50, ge=1, le=100, description="Max records to retrieve per page"
    ),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    store: VoiceStore = Depends(get_store),
):
    """
    Retrieves a paginated, filtered catalog of voice profiles from the persistence layer.

    Architectural Note:
    This endpoint acts as the primary data feed for UI lists and external integrations.
    Unlike the detail endpoint, it strictly returns 'VoiceSummary' objects. This DTO pattern
    strips away heavy data (like vectors or long prompts) to minimize network latency
    and serialization overhead during bulk retrieval.

    Filtering Strategy:
    Filtering is currently applied in-memory after fetching profiles from the store.
    While this is performant for datasets up to ~10k records (typical for this domain),
    future iterations might push these predicates down to the database query layer
    if the dataset grows significantly.

    Args:
        lang (Optional[str]): ISO language code filter (e.g., 'es'). Exact match.
        tag (Optional[str]): Semantic tag filter. Case-insensitive containment check.
        source_type (Optional[str]): Origin filter ('common', 'design', 'cloned').
        limit (int): Pagination limit to control response payload size.
        offset (int): Pagination offset for infinite scrolling implementations.
        store (VoiceStore): Injected singleton of the persistence controller.

    Returns:
        List[VoiceSummary]: An ordered list of lightweight voice representations,
        sorted by creation date (newest first).
    """
    all_profiles = store.get_all()
    filtered_profiles = []

    for profile in all_profiles:
        if lang and profile.language.lower() != lang.lower():
            continue

        if tag:
            profile_tags = [t.lower() for t in profile.tags]
            if tag.lower() not in profile_tags:
                continue

        if source_type and str(profile.source_type) != source_type:
            continue

        filtered_profiles.append(profile)

    filtered_profiles.sort(key=lambda x: x.created_at, reverse=True)

    paginated_profiles = filtered_profiles[offset : offset + limit]

    return [
        VoiceSummary(
            id=p.id,
            name=p.name,
            language=p.language,
            tags=p.tags,
            source_type=p.source_type,  
            created_at=p.created_at,
            description=p.description,
            preview_url=f"/v1/voices/{p.id}/audio",
        )
        for p in paginated_profiles
    ]


@router.get("/search", response_model=List[VoiceSummary])
def search_voices(
    q: str = Query(..., min_length=2, description="Natural language search query"),
    limit: int = Query(10, ge=1, le=50, description="Max relevant results"),
    store: VoiceStore = Depends(get_store),
    embed_engine: Any = Depends(get_embed_engine),
):
    """
    Performs a semantic similarity search using high-dimensional vector embeddings.

    Mechanism:
    1. The user's text query is passed to the injected Embedding Engine (Transformer model).
    2. The engine generates a dense vector representation of the query intent.
    3. This vector is compared against the pre-indexed 'semantic_embedding' of all voices
       using Cosine Similarity.

    Dependency Note:
    This endpoint triggers the 'Lazy Loading' of the ML engine if it hasn't been used yet.
    This design decision offloads the heavy memory cost of PyTorch/ONNX to the first
    actual search request, rather than slowing down the API startup.

    Args:
        q (str): The search text (e.g., "A deep, scary narrator for a horror movie").
        limit (int): Cap on the number of results to ensure relevance.
        store (VoiceStore): Injected persistence layer.
        embed_engine (Any): Injected ML inference engine wrapper.

    Returns:
        List[VoiceSummary]: Voices ranked by their semantic proximity to the query.

    Raises:
        HTTPException(503): If the Embedding Engine fails to initialize (e.g., missing weights).
        HTTPException(500): If the vector inference process crashes.
    """
    if not embed_engine:
        logger.error(
            "Semantic search requested but the Embedding Engine is not available."
        )
        raise HTTPException(
            status_code=503,
            detail="Search service unavailable due to initialization failure.",
        )

    try:
        query_vector = embed_engine.generate_embedding(q)
    except Exception as e:
        logger.error(f"Vector generation failed for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Failed to process search query.")

    results = store.search_semantic(query_vector, limit=limit)

    return [
        VoiceSummary(
            id=p.id,
            name=p.name,
            language=p.language,
            tags=p.tags,
            source_type=p.source_type,
            created_at=p.created_at,
            description=p.description,
            preview_url=f"/v1/voices/{p.id}/audio",
        )
        for p in results
    ]


@router.get("/voices/{voice_id}", response_model=VoiceDetail)
def get_voice_detail(voice_id: str, store: VoiceStore = Depends(get_store)):
    """
    Retrieves the complete technical specification for a single voice profile.

    This endpoint acts as the Single Source of Truth (SSOT) for consuming a voice.
    Unlike the summary endpoints, it exposes the 'anchor_text' (critical for
    In-Context Learning/Cloning) and the specific generation parameters used
    to create the voice.

    HATEOAS Implementation:
    The response includes an 'endpoints' object providing pre-constructed URLs
    to binary assets (audio, bundle, text). This decouples the client from knowing
    the API's internal URL routing structure, allowing backend refactors without
    breaking client implementations.

    Args:
        voice_id (str): UUID of the target voice.
        store (VoiceStore): Injected persistence layer.

    Returns:
        VoiceDetail: Full metadata profile + Asset Discovery Links.

    Raises:
        HTTPException(404): If the voice ID does not exist in the index.
    """
    profile = store.get_profile(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")

    return VoiceDetail(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        anchor_text=profile.anchor_text,
        language=profile.language,
        tags=profile.tags,
        source_type=profile.source_type,
        created_at=profile.created_at,
        params=profile.parameters,
        endpoints=VoiceEndpoints(
            audio=f"/v1/voices/{profile.id}/audio",
            bundle=f"/v1/voices/{profile.id}/bundle",
            text=f"/v1/voices/{profile.id}/text",
        ),
    )


# --- ASSET DELIVERY ENDPOINTS ---


@router.get("/voices/{voice_id}/audio")
def get_voice_audio(
    voice_id: str, store: VoiceStore = Depends(get_store), config=Depends(get_config)
):
    """
    Streams the raw audio reference file (Anchor) for a specific voice.

    Security & Resolution:
    This endpoint resolves the logical 'voice_id' to a physical file path stored
    in the secured assets directory. It validates that the file actually exists
    on the disk to prevent 500 errors during streaming. This abstraction prevents
    Path Traversal attacks by not accepting file paths directly from the user.

    MIME Type:
    Served as 'audio/wav'. This allows direct playback in HTML5 <audio> tags
    and seamless integration with TTS inference engines that require WAV input.

    Args:
        voice_id (str): UUID of the target voice.
        store (VoiceStore): Injected persistence layer.
        config (AppConfig): Configuration object for resolving the assets root path.

    Returns:
        FileResponse: Binary stream of the audio file.
    """
    profile = store.get_profile(voice_id)
    if not profile or not profile.anchor_audio_path:
        raise HTTPException(status_code=404, detail="Audio asset path is undefined.")

    file_path = os.path.join(config.paths.assets_dir, profile.anchor_audio_path)

    if not os.path.exists(file_path):
        logger.error(
            f"Data Integrity Error: File for voice {voice_id} missing at {file_path}"
        )
        raise HTTPException(
            status_code=404, detail="Physical audio file missing on server."
        )

    return FileResponse(file_path, media_type="audio/wav")


@router.get("/voices/{voice_id}/text")
def get_voice_text(voice_id: str, store: VoiceStore = Depends(get_store)):
    """
    Retrieves the canonical reference text for the voice.

    Why JSON?
    Returning raw text can lead to encoding issues or ambiguity with line breaks
    in some HTTP clients. Wrapping the text in a JSON object `{"text": "..."}`
    ensures strict UTF-8 compliance and makes it easier for frontend clients
    to parse and bind the data to UI components.

    Args:
        voice_id (str): UUID of the target voice.
        store (VoiceStore): Injected persistence layer.

    Returns:
        JSONResponse: Object containing the anchor text.
    """
    profile = store.get_profile(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice not found")

    return JSONResponse(content={"text": profile.anchor_text})


@router.get("/voices/{voice_id}/bundle")
def download_voice_bundle(
    voice_id: str, store: VoiceStore = Depends(get_store), config=Depends(get_config)
):
    """
    Dynamically packs and streams a portable Voice Bundle (.rnb).

    Operation:
    This is a compute-bound operation that:
    1. Fetches the latest metadata from the DB.
    2. Reads the physical audio file from disk.
    3. Serializes both into a ZIP archive structure in memory.

    Use Case:
    Essential for migrating voices between Resona instances (e.g., Dev -> Prod)
    or creating backups. By generating the bundle on-the-fly, we guarantee
    that the export always represents the exact current state of the voice,
    including any recent metadata edits.

    Args:
        voice_id (str): UUID of the target voice.
        store (VoiceStore): Injected persistence layer.
        config (AppConfig): Configuration object for resolving paths.

    Returns:
        StreamingResponse: Downloadable ZIP stream with proper Content-Disposition headers.
    """
    profile = store.get_profile(voice_id)
    if not profile or not profile.anchor_audio_path:
        raise HTTPException(status_code=404, detail="Voice asset invalid.")

    file_path = os.path.join(config.paths.assets_dir, profile.anchor_audio_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Source audio file missing.")

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        bundle_bytes = VoiceBundleIO.pack_bundle(profile.model_dump(), audio_bytes)

        safe_filename = f"{profile.name.replace(' ', '_')}_{voice_id[:6]}.rnb"

        return StreamingResponse(
            io.BytesIO(bundle_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    except Exception as e:
        logger.error(f"Bundle generation failed for {voice_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate voice bundle.")

@router.get("/emotions", response_model=List[EmotionSummary])
def list_emotions(
    lang: str = Query("en", description="ISO language code for UI localization (e.g., 'es', 'en')"),
    manager: EmotionManager = Depends(get_emotion_manager)
):
    """
    Retrieves the catalog of available emotional prosody profiles.
    
    The response automatically maps the display names to the requested language
    falling back to the canonical ID if a translation is missing.
    """
    vocab = manager.get_localized_vocab(lang=lang)["emotions"]
    results = []
    
    for eid, data in manager.catalog.get("emotions", {}).items():
        results.append(
            EmotionSummary(
                id=eid,
                name=vocab.get(eid, eid),
                modifiers=data.get("modifiers", {})
            )
        )
        
    return sorted(results, key=lambda x: x.name.lower())


@router.get("/intensifiers", response_model=List[IntensifierSummary])
def list_intensifiers(
    lang: str = Query("en", description="ISO language code for UI localization"),
    manager: EmotionManager = Depends(get_emotion_manager)
):
    """
    Retrieves the catalog of available prosody intensifiers.
    """
    vocab = manager.get_localized_vocab(lang=lang)["intensifiers"]
    results = []
    
    for iid, data in manager.catalog.get("intensifiers", {}).items():
        results.append(
            IntensifierSummary(
                id=iid,
                name=vocab.get(iid, iid),
                multiplier=data.get("multiplier", 1.0)
            )
        )
        
    return sorted(results, key=lambda x: x.name.lower())

@router.get("/emotions/{emotion_id}", response_model=EmotionSummary)
def get_emotion(
    emotion_id: str,
    lang: str = Query("en", description="ISO language code for UI localization"),
    manager: EmotionManager = Depends(get_emotion_manager)
):
    """
    Retrieves a specific emotional prosody profile by its canonical ID.
    
    Args:
        emotion_id (str): The canonical ID of the emotion.
        lang (str): The target language for the 'name' field.
        manager (EmotionManager): Injected singleton of the emotion controller.

    Returns:
        EmotionSummary: The localized emotion data.

    Raises:
        HTTPException: If the emotion_id does not exist in the catalog.
    """
    emotion_data = manager.catalog.get("emotions", {}).get(emotion_id)
    if not emotion_data:
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion_id}' not found.")
        
    vocab = manager.get_localized_vocab(lang=lang)["emotions"]
    
    return EmotionSummary(
        id=emotion_id,
        name=vocab.get(emotion_id, emotion_id),
        modifiers=emotion_data.get("modifiers", {})
    )


@router.get("/intensifiers/{intensifier_id}", response_model=IntensifierSummary)
def get_intensifier(
    intensifier_id: str,
    lang: str = Query("en", description="ISO language code for UI localization"),
    manager: EmotionManager = Depends(get_emotion_manager)
):
    """
    Retrieves a specific prosody intensifier by its canonical ID.
    
    Args:
        intensifier_id (str): The canonical ID of the intensifier.
        lang (str): The target language for the 'name' field.
        manager (EmotionManager): Injected singleton of the emotion controller.

    Returns:
        IntensifierSummary: The localized intensifier data.

    Raises:
        HTTPException: If the intensifier_id does not exist in the catalog.
    """
    intensifier_data = manager.catalog.get("intensifiers", {}).get(intensifier_id)
    if not intensifier_data:
        raise HTTPException(status_code=404, detail=f"Intensifier '{intensifier_id}' not found.")
        
    vocab = manager.get_localized_vocab(lang=lang)["intensifiers"]
    
    return IntensifierSummary(
        id=intensifier_id,
        name=vocab.get(intensifier_id, intensifier_id),
        multiplier=intensifier_data.get("multiplier", 1.0)
    )
    
@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Consults the current state of a synthesis job.
    
    The client should poll this endpoint until status is 'completed' or 'failed'.
    """
    task = ACTIVE_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found.")
        
    download_url = f"/v1/tasks/{task_id}/audio" if task["status"] == "completed" else None
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        download_url=download_url,
        error=task.get("error")
    )


@router.get("/tasks/{task_id}/audio")
def download_task_audio(
    task_id: str, 
    background_tasks: BackgroundTasks
):
    """
    Streams the generated audio file and triggers an automatic cleanup.
    
    Architectural Note:
    Once the stream is finished, a BackgroundTask is triggered to delete 
    the temporary file from the disk, ensuring the server remains stateless.
    """
    task = ACTIVE_TASKS.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(status_code=404, detail="Audio not ready or task not found.")

    file_path = task.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Physical audio file missing on server.")

    background_tasks.add_task(os.remove, file_path)

    return FileResponse(file_path, media_type="audio/wav")

@router.post("/voices/{voice_id}/synthesize", status_code=202)
async def enqueue_standard_synthesis(
    voice_id: str,
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    store: VoiceStore = Depends(get_store),
    config = Depends(get_config),
    provider = Depends(get_tts_provider)
):
    """
    Enqueues a standard text-to-speech task.
    Returns a task_id for polling.
    """
    if not provider:
        raise HTTPException(status_code=503, detail="TTS Engine is not initialized.")
        
    profile = store.get_profile(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice not found.")

    task_id = str(uuid.uuid4())
    ACTIVE_TASKS[task_id] = {"status": "pending", "file_path": None}
    
    target_lang = request.language or profile.language
    
    background_tasks.add_task(
        _process_synthesis_task, 
        task_id, voice_id, request.text, target_lang, None, store, config, provider
    )
    
    return {"task_id": task_id, "status": "pending"}


@router.post("/voices/{voice_id}/synthesize/emotion", status_code=202)
async def enqueue_emotional_synthesis(
    voice_id: str,
    request: EmotionSynthesisRequest,
    background_tasks: BackgroundTasks,
    store: VoiceStore = Depends(get_store),
    manager: EmotionManager = Depends(get_emotion_manager),
    config = Depends(get_config),
    provider = Depends(get_tts_provider)
):
    """
    Enqueues an emotional prosody synthesis task.
    Scales model parameters based on the requested emotion and intensifier.
    """
    if not provider:
        raise HTTPException(status_code=503, detail="TTS Engine is not initialized.")
        
    profile = store.get_profile(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice not found.")

    params = manager.calculate_parameters(request.emotion_id, request.intensifier_id)
    
    task_id = str(uuid.uuid4())
    ACTIVE_TASKS[task_id] = {"status": "pending", "file_path": None}
    
    target_lang = request.language or profile.language

    background_tasks.add_task(
        _process_synthesis_task, 
        task_id, voice_id, request.text, target_lang, params, store, config, provider
    )
    
    return {"task_id": task_id, "status": "pending"}

@router.post("/dialogs/validate")
def validate_dialog_template(
    payload: Dict[str, Any],
    store: VoiceStore = Depends(get_store),
    manager: EmotionManager = Depends(get_emotion_manager)
):
    """
    Evaluates a dialog script payload for both structural integrity and system resource availability.

    This endpoint operates as a dry-run validation barrier, ensuring that external API requests 
    are fully actionable before they are dispatched to the asynchronous orchestration queues or 
    reach the GPU level. The validation process is executed in a two-phase pipeline.

    Phase 1 (Structural Validation):
    The payload is serialized into a string and evaluated against the DialogScript factory method. 
    This phase strictly verifies schema compliance, confirming the presence of mandatory fields, 
    the correct typing of arrays, and the uniqueness of temporal execution indices across all lines.
    Failures in this phase result in immediate HTTP exceptions (400 or 422).

    Phase 2 (Resource Validation):
    Upon structural confirmation, the endpoint extracts all referenced canonical IDs for voices, 
    emotions, and acoustic intensifiers utilized within the timeline. It subsequently queries the 
    injected singleton instances of the VoiceStore and EmotionManager to verify physical availability.
    If requested resources are absent from the host system, the endpoint gracefully returns a detailed 
    audit payload mapping the exact missing dependencies without raising HTTP faults, enabling the 
    client application to prompt targeted corrections.

    Args:
        payload (Dict[str, Any]): The raw JSON dictionary representing the dialog configuration.
        store (VoiceStore): Injected persistence controller managing voice identity profiles.
        manager (EmotionManager): Injected logic controller mapping emotional prosody states.

    Returns:
        Dict[str, Any]: A structured audit response indicating boolean validity. If invalid due 
                        to missing dependencies, it provides a categorized manifest of unavailable 
                        canonical identifiers.

    Raises:
        HTTPException (400): If the payload exhibits logical structural violations such as duplicate indices.
        HTTPException (422): If the payload violates strict Pydantic schema typing definitions.
    """
    try:
        json_string = json.dumps(payload)
        clean_data = DialogScript.validate_template(json_string)
        
        missing_voices = []
        missing_emotions = []
        missing_intensities = []
        
        used_voices = set()
        used_emotions = set()
        used_intensities = set()
        
        for line in clean_data.get("script", []):
            if "voice_id" in line:
                used_voices.add(line["voice_id"])
            if "emotion" in line and line["emotion"]:
                used_emotions.add(line["emotion"])
            if "intensity" in line and line["intensity"]:
                used_intensities.add(line["intensity"])
                
        for vid in used_voices:
            if not store.get_profile(vid):
                missing_voices.append(vid)
                
        for eid in used_emotions:
            if eid not in manager.catalog.get("emotions", {}):
                missing_emotions.append(eid)
                
        for iid in used_intensities:
            if iid not in manager.catalog.get("intensifiers", {}):
                missing_intensities.append(iid)
                
        if missing_voices or missing_emotions or missing_intensities:
            return {
                "is_valid": False,
                "message": "Template is structurally valid, but missing required resources.",
                "missing_resources": {
                    "voices": missing_voices,
                    "emotions": missing_emotions,
                    "intensities": missing_intensities
                }
            }
            
        return {
            "is_valid": True,
            "message": "The template is structurally valid and all resources are available."
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError:
        raise HTTPException(status_code=422, detail="Schema validation failed.")
    
    import time
from pathlib import Path

@router.post("/dialogs/generate", status_code=202, response_model=DialogGenerationResponse)
def generate_dialog_project(
    payload: Dict[str, Any],
    ttl_hours: int = Query(24, ge=1, le=720, description="Time-to-live in hours before garbage collection."),
    config: Any = Depends(get_config),
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Initializes an asynchronous dialog synthesis process.

    This endpoint maps directly to the UI's import logic. It explicitly generates 
    the validated DialogScript definition first, leverages its identifier to construct 
    the physical workspace boundary, and finally maps the initialized states into 
    the overarching DialogProject entity before dispatching the background worker.

    Args:
        payload (Dict[str, Any]): The dictionary mapping of the dialog script.
        ttl_hours (int): The maximum lifespan of the generated assets in hours.
        config (Any): Injected global application configuration for path resolution.
        orchestrator (Any): Injected service delegating detached OS-level workers.

    Returns:
        DialogGenerationResponse: An acknowledgment envelope containing the allocated 
                                project identifier, expiration timestamp, and line mappings.

    Raises:
        HTTPException (400): If the inner dictionary structure violates domain logic integrity.
        HTTPException (422): If the schema evaluation fails strict typing.
        HTTPException (500): If filesystem allocation or database persistence operations fail.
    """
    try:
        json_string = json.dumps(payload)
        script = DialogScript.import_template(json_string)
        workspace_dir = str(Path(config.paths.dialogspace_dir) / script.id)
        expiration_timestamp = time.time() + (ttl_hours * 3600)
        
        project = DialogProject(
            source=ProjectSource.API,
            definition=script,
            states=[
                LineState(line_id=line.id, index=line.index, status=LineStatus.PENDING) 
                for line in script.script
            ],
            project_path=workspace_dir,
            status=ProjectStatus.IDLE,
            expires_at=expiration_timestamp
        )
        
        orchestrator.add_project(project)
        orchestrator.start_generation(project.id)
        
        return DialogGenerationResponse(
            project_id=project.id,
            expires_at=project.expires_at,
            message="Generation task successfully dispatched to the background worker.",
            lines=[
                LineIdentifier(index=state.index, line_id=state.line_id) 
                for state in project.states
            ]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError:
        raise HTTPException(status_code=422, detail="Schema validation failed.")
    except Exception as e:
        logger.error(f"Generation initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during project initialization.")


@router.get("/dialogs/{project_id}/status", response_model=DialogStatusResponse)
def get_dialog_project_status(
    project_id: str,
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Retrieves the highly optimized real-time execution status of a project.

    This endpoint acts as the primary polling target. It returns a lightweight 
    payload containing the execution state, progress metrics, and line progression, 
    deliberately omitting the heavy script definition to minimize bandwidth 
    during high-frequency client polling.

    Args:
        project_id (str): The canonical identifier of the target dialog project.
        orchestrator (Any): Injected service providing safe data access methods.

    Returns:
        DialogStatusResponse: A lean status envelope with progress metrics, output paths, 
                              and a granular tracker array for individual lines.

    Raises:
        HTTPException (404): If the provided project_id does not exist.
    """
    project_data = orchestrator.get_project_data_safe(project_id)
    
    if not project_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Dialog project with ID {project_id} not found."
        )
    
    states = project_data.get("states", [])
    total_lines = len(states)
    completed_lines = sum(1 for state in states if state.get("status") == "COMPLETED")
    
    return DialogStatusResponse(
        project_id=project_data.get("id"),
        status=project_data.get("status"),
        progress=DialogProgress(
            total_lines=total_lines,
            completed_lines=completed_lines,
            pending_lines=total_lines - completed_lines
        ),
        outputs=DialogOutputs(
            merged_audio_path=project_data.get("merged_audio_path"),
            merged_mp3_path=project_data.get("merged_mp3_path")
        ),
        line_states=[
            LineStatusTracker(
                index=state.get("index"),
                line_id=state.get("line_id"),
                status=state.get("status"),
                audio_path=state.get("audio_path"),
                error=state.get("error")
            )
            for state in states
        ]
    )
    
@router.post("/dialogs/{project_id}/merge", response_model=DialogMergeResponse)
def merge_dialog_project(
    project_id: str,
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Assembles the completed individual dialogue lines into a master audio file.

    This endpoint triggers the AudioEngine via the orchestrator to process all 
    individual audio segments associated with the project. It strictly enforces 
    a state validation, ensuring that timeline generation has successfully concluded 
    before attempting to compute the mix.

    Args:
        project_id (str): The canonical identifier of the target dialog project.
        orchestrator (Any): Injected service delegating the heavy audio mixing.

    Returns:
        DialogMergeResponse: An envelope containing the paths to the finalized assets.

    Raises:
        HTTPException (400): If the project timeline is not in a COMPLETED state.
        HTTPException (404): If the provided project_id does not exist.
        HTTPException (500): If the physical audio mix or database update fails.
    """
    project_data = orchestrator.get_project_data_safe(project_id)
    
    if not project_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Dialog project with ID {project_id} not found."
        )
        
    current_status = project_data.get("status")
    if current_status not in (ProjectStatus.COMPLETED.value, "COMPLETED"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot merge project. Current status is '{current_status}'. All lines must be COMPLETED before assembly."
        )

    try:
        updated_path = orchestrator.merge_project_audio(project_id)
        
        if not updated_path:
            raise HTTPException(
                status_code=500, 
                detail="Merge operation failed to return updated persistence data."
            )
            
        updated_data = orchestrator.get_project_data_safe(project_id)
            
        return DialogMergeResponse(
            project_id=updated_data.get("id"),
            message="Master audio timeline successfully assembled.",
            merged_audio_path=updated_data.get("merged_audio_path"),
            merged_mp3_path=updated_data.get("merged_mp3_path")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Audio merge failed for project {project_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during audio assembly."
        )

@router.get("/dialogs/{project_id}/download", response_class=FileResponse)
def download_dialog_assets(
    project_id: str,
    format: str = Query("wav", pattern="^(wav|mp3|bundle)$", description="Asset format."),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Dispatches generated dialogue assets utilizing absolute filesystem resolution.

    This implementation provides dynamic distribution of project artifacts. For master 
    files, it resolves paths by joining the project's workspace root with the relative 
    paths stored in the database. For bundles, it generates a ZIP archive on-the-fly 
    within the system's temporary directory, ensuring the project workspace remains 
    unpolluted, while scheduling an atomic cleanup of the transient ZIP post-transfer.

    Args:
        project_id (str): The unique identifier for the dialog project.
        format (str): Requested asset format (wav, mp3, or bundle).
        background_tasks (BackgroundTasks): Utility for post-response cleanup.
        orchestrator (Any): Service providing safe access to project metadata.

    Returns:
        FileResponse: High-performance binary stream of the requested asset.

    Raises:
        HTTPException (404): If the project metadata or physical files are missing.
        HTTPException (500): If the archival process encounters a system-level error.
    """
    project_data = orchestrator.get_project_data_safe(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found.")

    workspace_path = project_data.get("project_path")
    if not workspace_path or not os.path.exists(workspace_path):
        raise HTTPException(status_code=404, detail="Project workspace directory missing on disk.")

    if format == "bundle":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            temp_zip_path = tmp.name
        
        try:
            shutil.make_archive(temp_zip_path.replace(".zip", ""), 'zip', root_dir=workspace_path)
            background_tasks.add_task(os.remove, temp_zip_path)
            
            return FileResponse(
                path=temp_zip_path,
                media_type="application/zip",
                filename=f"bundle_{project_id}.zip"
            )
        except Exception as e:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            logger.error(f"On-the-fly bundle assembly failed for {project_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to assemble the project bundle.")

    target_field = "merged_mp3_path" if format == "mp3" else "merged_audio_path"
    relative_file_path = project_data.get(target_field)

    if not relative_file_path:
        raise HTTPException(status_code=404, detail=f"The {format.upper()} asset has not been generated.")

    full_path = os.path.join(workspace_path, relative_file_path)

    if not os.path.exists(full_path):
        logger.error(f"Integrity error: Expected file at {full_path} is missing.")
        raise HTTPException(status_code=404, detail="Physical asset missing on server.")

    return FileResponse(
        path=full_path,
        media_type="audio/mpeg" if format == "mp3" else "audio/wav",
        filename=os.path.basename(full_path)
    )
    
@router.get("/dialogs/{project_id}/lines/{line_id}/audio", response_class=FileResponse)
def download_line_audio(
    project_id: str,
    line_id: str,
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Retrieves the synthesized audio segment for a specific dialogue line.

    This endpoint facilitates incremental asset consumption by allowing clients 
    to retrieve individual WAV segments as soon as their synthesis status 
    reaches the COMPLETED state. It performs a dual-layer validation, verifying 
    both the logical metadata in the persistence layer and the physical 
    integrity of the asset on the filesystem.

    Args:
        project_id (str): The canonical identifier of the dialog project.
        line_id (str): The specific identifier of the dialogue line within the script.
        orchestrator (Any): Injected service for safe metadata and state retrieval.

    Returns:
        FileResponse: The binary stream of the individual audio segment.

    Raises:
        HTTPException (400): If the requested line has not reached the COMPLETED status.
        HTTPException (404): If the project, the line identifier, or the physical file is missing.
    """
    project_data = orchestrator.get_project_data_safe(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found.")

    states = project_data.get("states", [])
    line_state = next((s for s in states if s.get("line_id") == line_id), None)

    if not line_state:
        raise HTTPException(
            status_code=404, 
            detail=f"Line '{line_id}' does not belong to project '{project_id}'."
        )

    if line_state.get("status") != LineStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400, 
            detail=f"Line audio is unavailable. Current status: '{line_state.get('status')}'."
        )

    relative_audio_path = line_state.get("audio_path")
    workspace_path = project_data.get("project_path")

    if not relative_audio_path or not workspace_path:
        raise HTTPException(
            status_code=404, 
            detail="Audio path metadata is missing for this completed line."
        )

    full_path = os.path.normpath(os.path.join(workspace_path, relative_audio_path))

    if not os.path.exists(full_path):
        logger.error(f"Asset integrity failure: Line audio missing at {full_path}")
        raise HTTPException(
            status_code=404, 
            detail="The physical audio file for this line is missing from the server."
        )

    return FileResponse(
        path=full_path,
        media_type="audio/wav",
        filename=f"line_{line_id}.wav"
    )
    
@router.delete("/dialogs/{project_id}", status_code=204)
def delete_dialog_project(
    project_id: str,
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Executes a complete and destructive purge of a dialog project and its resources.

    This operation implements a multi-tier cleanup strategy to ensure system 
    integrity. It first triggers an OS-level process interruption to terminate 
    any active background synthesis workers associated with the project identifier, 
    preventing file descriptor leaks. Subsequently, it performs a recursive 
    annihilation of the physical workspace directory. Finally, it purges the 
    project metadata from the persistence layer to ensure absolute removal 
    from the database records.

    Args:
        project_id (str): The canonical identifier of the project to be annihilated.
        orchestrator (Any): Injected service providing lifecycle and process management.

    Raises:
        HTTPException (404): If the project identifier does not exist in the database.
        HTTPException (500): If the system fails to kill active processes or clear disk assets.
    """
    project_data = orchestrator.get_project_data_safe(project_id)
    
    if not project_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Project with ID {project_id} not found."
        )

    try:
        orchestrator.purge_project_assets(project_id)
        
        orchestrator.delete_project(project_id)
        
        return
        
    except Exception as e:
        logger.error(f"Critical failure during destructive cleanup for project {project_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during the project purging sequence."
        )
        
@router.get("/dialogs", response_model=ProjectListResponse)
def list_dialog_projects(
    source: str = Query(
        "all", 
        pattern="^(ui|api|all)$", 
        description="Filters project discovery by creation source ('ui', 'api', or 'all')."
    ),
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Retrieves an inventory of dialog projects with source-based discrimination.

    This discovery endpoint interfaces with the orchestration layer to extract the 
    global project registry. It supports dynamic segmentation, allowing consumers 
    to isolate projects initiated via the administrative dashboard, the REST API, 
    or retrieve the entire collection. The response is structured for lightweight 
    consumption in list-view components.

    Args:
        source (str): Source filter discriminator ('ui', 'api', or 'all').
        orchestrator (Any): Injected service for registry and metadata access.

    Returns:
        ProjectListResponse: An envelope containing project count and metadata list.

    Raises:
        HTTPException (500): If the persistence layer fails to resolve the registry.
    """
    try:
        query_source = None
        if source != "all":
            query_source = ProjectSource(source)
            
        raw_projects = orchestrator.get_all_projects(source=query_source)
        
        project_list = [
            ProjectListItem(
                project_id=data.get("id"),
                name=data.get("definition", {}).get("name", "Unnamed Project"),
                status=data.get("status"),
                source=data.get("source", ProjectSource.API.value),
                created_at=data.get("definition", {}).get("created_at")
            )
            for data in raw_projects
        ]
        
        return ProjectListResponse(
            total=len(project_list),
            projects=project_list
        )
        
    except Exception as e:
        logger.error(f"Failed to resolve project inventory for source '{source}': {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while resolving project inventory."
        )

@router.get("/dialogs/{project_id}/export")
def export_dialog_script(
    project_id: str,
    orchestrator: Any = Depends(get_orchestrator)
):
    """
    Exports a project script as a portable JSON template.

    This endpoint retrieves the raw project data from the persistence layer and 
    instantiates the DialogProject model to leverage its internal template 
    export logic. The resulting JSON is stripped of instance-specific 
    identifiers and execution states, then delivered as a downloadable 
    file attachment.

    Args:
        project_id (str): The unique identifier of the project to export.
        orchestrator (Any): Injected service for safe data retrieval.

    Returns:
        Response: A JSON payload configured as a file download.

    Raises:
        HTTPException (404): If the project data cannot be found.
    """
    data = orchestrator.get_project_data_safe(project_id)
    
    if not data:
        raise HTTPException(
            status_code=404, 
            detail=f"Project with ID {project_id} not found."
        )
    
    project = DialogProject(**data)
    
    json_template = project.definition.export_as_template()
    
    safe_filename = project.definition.name.replace(" ", "_").lower()
    
    return Response(
        content=json_template,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={safe_filename}_template.json"
        }
    )