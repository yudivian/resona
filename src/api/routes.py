import os
import io
import logging
import uuid
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi import BackgroundTasks 

from src.models import (
    VoiceSummary,
    VoiceDetail,
    VoiceEndpoints,
    EmotionSummary,
    IntensifierSummary,
    TaskStatusResponse,
    SynthesisRequest, 
    EmotionSynthesisRequest
)

from src.backend.store import VoiceStore
from src.backend.io import VoiceBundleIO
from src.api.dependencies import (
    get_store,
    get_embed_engine,
    get_config,
    get_emotion_manager,
    get_tts_provider
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