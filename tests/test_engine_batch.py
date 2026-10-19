import sys
import os
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.backend.store import VoiceStore
from src.backend.engine import TTSModelProvider, InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main() -> None:
    """
    Executes an isolated integration test for the InferenceEngine's batch processing capabilities.
    """
    logging.info("Starting isolated batch inference integration test.")
    
    voice_store = VoiceStore(settings)
    profiles = voice_store.get_all()
    
    if not profiles:
        raise RuntimeError("No voice profiles found in the database. Please create at least one voice profile before testing.")
        
    target_voice_id = sys.argv[1] if len(sys.argv) > 1 else profiles[0].id
    voice_profile = voice_store.get_profile(target_voice_id)
    
    if not voice_profile:
        raise ValueError(f"Target voice profile ID '{target_voice_id}' could not be located in the database.")
        
    logging.info(f"Acquired target Voice Profile: {voice_profile.name} (ID: {voice_profile.id})")
    
    output_dir = Path("tests/batch_output_temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory ensured at absolute path: {output_dir.absolute()}")
    
    logging.info("Initializing TTSModelProvider and allocating neural network weights into VRAM...")
    provider = TTSModelProvider(settings)
    
    target_language = voice_profile.language if voice_profile.language else "es"
    logging.info(f"Instantiating InferenceEngine configured for language: {target_language}")
    
    engine = InferenceEngine(
        config=settings,
        tts_provider=provider,
        lang=target_language
    )
    
    logging.info("Injecting identity embedding and deterministic seed into the engine state...")
    engine.load_identity_from_state(
        vector=voice_profile.identity_embedding,
        seed=voice_profile.seed
    )
    
    texts = [
        "Esta es la primera línea del procesamiento por lotes, verificando la inicialización.",
        "Esta es la segunda línea, diseñada para comprobar que la GPU procesa múltiples tensores de forma concurrente.",
        "Y esta es la tercera línea, confirmando que la identidad de voz se ha clonado correctamente para todo el bloque."
    ]
    
    standard_paths = [str(output_dir / f"standard_batch_output_{i}.wav") for i in range(len(texts))]
    
    logging.info("Commencing standard batch inference forward pass on the GPU...")
    start_time = time.time()
    
    engine.render_batch(texts=texts, output_paths=standard_paths)
    
    standard_duration = time.time() - start_time
    logging.info(f"Standard batch inference completed successfully in {standard_duration:.2f} seconds.")
    
    emotion_params = {"temp": 0.45, "top_p": 0.85, "penalty": 1.15}
    emotion_paths = [str(output_dir / f"emotion_batch_output_{i}.wav") for i in range(len(texts))]
    
    logging.info(f"Commencing emotion-driven batch inference with injected parameters: {emotion_params}...")
    start_time = time.time()
    
    engine.render_batch_with_emotion(texts=texts, emotion_params=emotion_params, output_paths=emotion_paths)
    
    emotion_duration = time.time() - start_time
    logging.info(f"Emotion-driven batch inference completed successfully in {emotion_duration:.2f} seconds.")
    
    logging.info("Initiating file system verification for all requested audio artifacts...")
    all_paths = standard_paths + emotion_paths
    
    for path_str in all_paths:
        file_path = Path(path_str)
        if file_path.exists() and file_path.stat().st_size > 0:
            logging.info(f"Integrity verified: {file_path.name} ({file_path.stat().st_size} bytes)")
        else:
            logging.error(f"Artifact verification failed. File is missing or empty: {path_str}")
            
    logging.info("Integration test concluded. Audio artifacts have been intentionally preserved in the output directory for manual acoustic review.")

if __name__ == "__main__":
    main()