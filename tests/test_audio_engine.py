"""
Standalone test suite for the AudioEngine.
Verifies tensor concatenation, resampling, and fading logic with real files.
"""

import logging
import sys
from pathlib import Path

# Adjusting path to allow execution from the project root
sys.path.append(str(Path(__file__).parent.parent))

from src.backend.audio_engine import AudioEngine, AudioSegmentConfig

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AudioEngineTest")

def run_folder_test(input_folder: str, output_filename: str = "test_merge_result.wav"):
    """
    Scans a directory for WAV files and merges them using the AudioEngine.
    
    Args:
        input_folder (str): Path to the directory containing source audio clips.
        output_filename (str): Name of the generated master file.
    """
    input_path = Path(input_folder)
    output_path = input_path / output_filename

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input path does not exist or is not a directory: {input_folder}")
        return

    # 1. Discover audio files (WAV)
    audio_files = sorted(list(input_path.glob("*.wav")))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {input_folder}")
        return

    logger.info(f"🚀 Starting Engine Test. Found {len(audio_files)} files in '{input_folder}'")

    # 2. Prepare agnostic configurations
    # We apply different settings to test the engine's flexibility
    segments = []
    for i, file_path in enumerate(audio_files):
        logger.info(f"  [Segment {i+1}] Preparing: {file_path.name}")
        
        config = AudioSegmentConfig(
            path=file_path,
            fade_in_ms=100,      # Slightly longer fade-in for testing
            fade_out_ms=100,     # Slightly longer fade-out for testing
            post_delay_ms=1000,  # 1 second gap between clips
            room_tone_level=0.0002
        )
        segments.append(config)

    # 3. Initialize Engine and Merge
    engine = AudioEngine(target_sample_rate=24000)
    
    logger.info("🛠️ Invoking AudioEngine.merge_segments...")
    
    success = engine.merge_segments(segments, output_path)

    # 4. Final Validation
    if success and output_path.exists():
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"✅ SUCCESS! Master file generated at: {output_path}")
        logger.info(f"📊 Final stats: {file_size_kb:.2f} KB | Path: {output_path.absolute()}")
    else:
        logger.error("❌ FAILURE: The engine could not generate the merged file. Check file permissions or tensor shapes.")

if __name__ == "__main__":
    # You can pass the folder as an argument or edit this string
    DEFAULT_TEST_FOLDER = "path/to/your/audio/folder" 
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = DEFAULT_TEST_FOLDER
        logger.info(f"No folder provided as argument. Using default: {target_dir}")

    run_folder_test(target_dir)